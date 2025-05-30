import streamlit as st

# 페이지 설정
st.set_page_config(page_title="KOMI 모니터링", layout="wide")

import json
import time
import numpy as np
import cv2
import base64
from datetime import datetime, timedelta
from PIL import Image
import asyncio
import aiohttp
import threading
from concurrent.futures import ThreadPoolExecutor
import queue
import collections
import random

# 서버 URL 설정
API_URL = "http://localhost:8000"

# Streamlit의 query parameter를 사용하여 서버 URL을 설정
if 'server_url' in st.query_params:
    API_URL = st.query_params['server_url']

# 스레드 안전 데이터 구조
image_queues = {}  # 카메라별 이미지 큐
is_running = True
selected_cameras = []

# 서버 시간 동기화 관련 변수
server_time_offset = 0.0  # 서버와의 시간차 (초 단위)
last_time_sync = 0  # 마지막 시간 동기화 시간
TIME_SYNC_INTERVAL = 300  # 5분으로 간격 증가

# 연결 재시도 설정
MAX_RECONNECT_ATTEMPTS = 5
RECONNECT_DELAY = 1.0  # 초 단위
connection_attempts = {}  # 카메라ID -> 시도 횟수

# 동기화 버퍼 설정
sync_buffer = {}  # 카메라 ID -> 최근 프레임 버퍼 (collections.deque)
SYNC_BUFFER_SIZE = 10  # 각 카메라별 버퍼 크기
MAX_SYNC_DIFF_MS = 100  # 프레임 간 최대 허용 시간 차이 (밀리초)

# WebSocket 연결 상태
ws_connection_status = {}  # 카메라ID -> 상태 ("connected", "disconnected", "reconnecting")

# 스레드별 전용 세션과 이벤트 루프
thread_local = threading.local()

# 이미지 처리를 위한 스레드 풀
thread_pool = ThreadPoolExecutor(max_workers=4)

# 세션 상태 초기화
if 'selected_cameras' not in st.session_state:
    st.session_state.selected_cameras = []
if 'cameras' not in st.session_state:
    st.session_state.cameras = []
if 'server_status' not in st.session_state:
    st.session_state.server_status = None
if 'camera_images' not in st.session_state:
    st.session_state.camera_images = {}
if 'image_update_time' not in st.session_state:
    st.session_state.image_update_time = {}
if 'sync_status' not in st.session_state:
    st.session_state.sync_status = "준비 중..."
if 'connection_status' not in st.session_state:
    st.session_state.connection_status = {}

# Base64 이미지 디코딩 함수
def decode_image(base64_image):
    """Base64 인코딩된 이미지를 디코딩하여 numpy 배열로 변환"""
    try:
        if not base64_image:
            return None
        img_data = base64.b64decode(base64_image)
        np_arr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"이미지 디코딩 오류: {str(e)}")
        return None

# 스레드별 세션 및 이벤트 루프 관리
def get_session():
    """현재 스레드의 세션 반환 (없으면 생성)"""
    if not hasattr(thread_local, "session"):
        thread_local.session = None
    return thread_local.session

def get_event_loop():
    """현재 스레드의 이벤트 루프 반환 (없으면 생성)"""
    if not hasattr(thread_local, "loop"):
        thread_local.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(thread_local.loop)
    return thread_local.loop

# 비동기 HTTP 클라이언트 세션 초기화
async def init_session():
    """비동기 세션 초기화 (스레드별)"""
    if not get_session():
        # 타임아웃 설정 추가
        timeout = aiohttp.ClientTimeout(total=10, connect=5, sock_connect=5, sock_read=5)
        thread_local.session = aiohttp.ClientSession(timeout=timeout)
    return thread_local.session

# 비동기 HTTP 클라이언트 세션 종료
async def close_session():
    """현재 스레드의 세션 닫기"""
    session = get_session()
    if session:
        await session.close()
        thread_local.session = None

# 동기 함수에서 비동기 작업 실행을 위한 헬퍼 함수
def run_async(coroutine):
    """동기 함수에서 비동기 코루틴 실행"""
    loop = get_event_loop()
    return loop.run_until_complete(coroutine)

# 비동기 카메라 목록 가져오기
async def async_get_cameras():
    """비동기적으로 카메라 목록 가져오기"""
    try:
        session = await init_session()
        # 타임아웃 파라미터를 숫자 대신 ClientTimeout 객체로 변경
        request_timeout = aiohttp.ClientTimeout(total=2)
        async with session.get(f"{API_URL}/cameras", timeout=request_timeout) as response:
            if response.status == 200:
                data = await response.json()
                return data.get("cameras", []), "연결됨"
            return [], "오류"
    except Exception as e:
        print(f"카메라 목록 요청 오류: {str(e)}")
        return [], "연결 실패"

# 동기 카메라 목록 가져오기 (Streamlit 호환용)
def get_cameras():
    """카메라 목록 가져오기 (동기 래퍼)"""
    cameras, status = run_async(async_get_cameras())
    st.session_state.server_status = status
    return cameras

# 스레드에서 이미지 디코딩 처리
def process_image_in_thread(image_data):
    """별도 스레드에서 이미지 처리"""
    try:
        if not image_data:
            return None
        if isinstance(image_data, str) and image_data.startswith('http'):  # URL인 경우 (향후 확장성)
            return None
        else:
            return decode_image(image_data)  # Base64 이미지인 경우
    except Exception as e:
        print(f"이미지 처리 오류: {str(e)}")
        return None

# 동기화 버퍼 초기화
def init_sync_buffer(camera_ids):
    """동기화 버퍼 초기화 함수"""
    global sync_buffer
    for camera_id in camera_ids:
        if camera_id not in sync_buffer:
            sync_buffer[camera_id] = collections.deque(maxlen=SYNC_BUFFER_SIZE)

# 동기화된 프레임 쌍 찾기
def find_synchronized_frames():
    """여러 카메라에서 타임스탬프가 가장 가까운 프레임 쌍 찾기"""
    global sync_buffer
    
    if len(st.session_state.selected_cameras) < 2:
        return None
    
    if not all(camera_id in sync_buffer and len(sync_buffer[camera_id]) > 0 
               for camera_id in st.session_state.selected_cameras):
        return None
    
    # 가장 최근의 타임스탬프 기준으로 시작
    latest_frame_times = {
        camera_id: max([frame["time"] for frame in sync_buffer[camera_id]])
        for camera_id in st.session_state.selected_cameras
    }
    
    # 가장 늦은 타임스탬프 찾기
    reference_time = min(latest_frame_times.values())
    
    # 각 카메라에서 기준 시간과 가장 가까운 프레임 찾기
    best_frames = {}
    for camera_id in st.session_state.selected_cameras:
        closest_frame = min(
            [frame for frame in sync_buffer[camera_id]],
            key=lambda x: abs((x["time"] - reference_time).total_seconds())
        )
        
        # 시간 차이가 허용 범위 내인지 확인
        time_diff_ms = abs((closest_frame["time"] - reference_time).total_seconds() * 1000)
        if time_diff_ms <= MAX_SYNC_DIFF_MS:
            best_frames[camera_id] = closest_frame
        else:
            return None
    
    # 모든 카메라에 대해 프레임을 찾았는지 확인
    if len(best_frames) == len(st.session_state.selected_cameras):
        # 동기화 상태 업데이트
        max_diff = max([abs((f["time"] - reference_time).total_seconds() * 1000) 
                        for f in best_frames.values()])
        st.session_state.sync_status = f"동기화됨 (최대 차이: {max_diff:.1f}ms)"
        return best_frames
    
    return None

# 서버 시간 동기화 함수 - 간소화 및 안정성 향상
async def sync_server_time():
    """서버 시간과 로컬 시간의 차이를 계산"""
    global server_time_offset, last_time_sync
    
    # 이미 최근에 동기화했다면 스킵
    current_time = time.time()
    if current_time - last_time_sync < TIME_SYNC_INTERVAL:
        return True
    
    try:
        session = await init_session()
        
        # 무작위 지연 추가 (서버 부하 분산)
        jitter = random.uniform(0, 1.0)
        await asyncio.sleep(jitter)
        
        local_time_before = time.time()
        # 타임아웃 파라미터를 숫자 대신 ClientTimeout 객체로 변경
        request_timeout = aiohttp.ClientTimeout(total=2)
        async with session.get(f"{API_URL}/server_time", timeout=request_timeout) as response:
            if response.status != 200:
                return False
                
            local_time_after = time.time()
            data = await response.json()
            
            server_timestamp = data.get("timestamp")
            if not server_timestamp:
                return False
            
            network_delay = (local_time_after - local_time_before) / 2
            local_time_avg = local_time_before + network_delay
            server_time_offset = server_timestamp - local_time_avg
            last_time_sync = time.time()
            return True
    except asyncio.TimeoutError:
        # 타임아웃은 조용히 처리
        return False
    except Exception:
        # 그 외 오류도 조용히 처리
        return False

# 서버 시간 기준 현재 시간 반환
def get_server_time():
    """서버 시간 기준의 현재 시간 계산"""
    return datetime.now() + timedelta(seconds=server_time_offset)

# 연결 상태 업데이트 함수
def update_connection_status(camera_id, status):
    """카메라 연결 상태 업데이트"""
    global ws_connection_status
    
    # 전역 상태 업데이트
    ws_connection_status[camera_id] = status
    
    # 백그라운드 스레드에서 Streamlit 세션 상태에 직접 접근하지 않음
    # 대신 image_queues를 통해 상태만 업데이트
    
    # 연결 시도 횟수 관리
    if status == "connected":
        connection_attempts[camera_id] = 0
    elif status == "disconnected":
        if camera_id not in connection_attempts:
            connection_attempts[camera_id] = 0

# 비동기 이미지 업데이트 함수
async def update_images():
    """백그라운드에서 이미지를 가져오는 함수 (WebSocket 기반)"""
    global selected_cameras, is_running, last_time_sync
    
    # 세션 초기화
    await init_session()
    
    # 초기 서버 시간 동기화
    await sync_server_time()
    
    # 카메라별 WebSocket 연결 태스크 저장
    stream_tasks = {}
    
    try:
        while is_running:
            # 주기적 서버 시간 동기화 - 현재 시간과 비교하여 판단
            if time.time() - last_time_sync >= TIME_SYNC_INTERVAL:
                await sync_server_time()
            
            # 전역 변수로 카메라 ID 목록 확인
            camera_ids = selected_cameras
            
            if camera_ids:
                # 동기화 버퍼 초기화
                init_sync_buffer(camera_ids)
                
                # 기존 스트림 중 필요없는 것 종료
                for camera_id in list(stream_tasks.keys()):
                    if camera_id not in camera_ids:
                        if not stream_tasks[camera_id].done():
                            stream_tasks[camera_id].cancel()
                        del stream_tasks[camera_id]
                
                # 새 카메라에 대한 WebSocket 스트림 시작
                for camera_id in camera_ids:
                    if camera_id not in stream_tasks or stream_tasks[camera_id].done():
                        # 재연결 시도할 때 약간의 지연 추가 (무작위)
                        jitter = random.uniform(0, 0.5)
                        await asyncio.sleep(jitter)
                        stream_tasks[camera_id] = asyncio.create_task(
                            connect_to_camera_stream(camera_id)
                        )
            
            # 요청 간격 조절
            await asyncio.sleep(0.1)
    except asyncio.CancelledError:
        # 정상적인 취소, 조용히 처리
        pass
    except Exception:
        # 다른 예외, 조용히 처리
        pass
    finally:
        # 모든 스트림 태스크 취소
        for task in stream_tasks.values():
            if not task.done():
                task.cancel()
        
        # 사용했던 세션 정리
        await close_session()

# 백그라운드 스레드에서 비동기 루프 실행
def run_async_loop():
    """비동기 루프를 실행하는 스레드 함수"""
    # 이 스레드 전용 이벤트 루프 생성
    loop = get_event_loop()
    
    try:
        # 이미지 업데이트 태스크 생성
        task = loop.create_task(update_images())
        
        # 이벤트 루프 실행
        loop.run_until_complete(task)
    except Exception as e:
        print(f"비동기 루프 오류: {str(e)}")
    finally:
        # 모든 태스크 취소
        for task in asyncio.all_tasks(loop):
            task.cancel()
        
        # 취소된 태스크 완료 대기
        if asyncio.all_tasks(loop):
            loop.run_until_complete(asyncio.gather(*asyncio.all_tasks(loop), return_exceptions=True))
        
        # 루프 중지
        loop.stop()

# WebSocket 연결 및 이미지 스트리밍 수신 - 안정성 개선
async def connect_to_camera_stream(camera_id):
    """WebSocket을 통해 카메라 스트림에 연결"""
    global connection_attempts
    
    # 연결 상태 업데이트
    update_connection_status(camera_id, "reconnecting")
    
    # 최대 재연결 시도 횟수 확인
    if camera_id in connection_attempts and connection_attempts[camera_id] >= MAX_RECONNECT_ATTEMPTS:
        # 지수 백오프 지연 계산
        delay = min(30, RECONNECT_DELAY * (2 ** connection_attempts[camera_id]))
        await asyncio.sleep(delay)
    
    # 재연결 시도 횟수 증가
    if camera_id not in connection_attempts:
        connection_attempts[camera_id] = 0
    connection_attempts[camera_id] += 1
    
    try:
        session = await init_session()
        # WebSocket URL 구성
        ws_url = f"{API_URL.replace('http://', 'ws://')}/ws/stream/{camera_id}"
        
        # 향상된 WebSocket 옵션
        heartbeat = 30.0  # 30초 핑/퐁
        ws_timeout = aiohttp.ClientWSTimeout(ws_close=60.0)  # WebSocket 종료 대기 시간 60초
        
        async with session.ws_connect(
            ws_url, 
            heartbeat=heartbeat,
            timeout=ws_timeout,
            max_msg_size=0,  # 무제한
            compress=False  # 웹소켓 압축 비활성화로 성능 향상
        ) as ws:
            # 연결 성공 - 상태 업데이트 및 시도 횟수 초기화
            update_connection_status(camera_id, "connected")
            connection_attempts[camera_id] = 0
            
            last_ping_time = time.time()
            ping_interval = 25  # 25초마다 핑 전송 (30초 하트비트보다 짧게)
            
            while is_running:
                # 핑 전송 (주기적으로) - 서버 핑/퐁 메커니즘과 별개로 유지
                current_time = time.time()
                if current_time - last_ping_time >= ping_interval:
                    try:
                        await ws.ping()
                        last_ping_time = current_time
                    except:
                        # 핑 실패 시 루프 탈출하여 재연결
                        break
                
                # 데이터 수신 (짧은 타임아웃으로 반응성 유지)
                try:
                    msg = await asyncio.wait_for(ws.receive(), timeout=0.1)
                    
                    if msg.type == aiohttp.WSMsgType.CLOSED:
                        break
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        break
                    elif msg.type == aiohttp.WSMsgType.TEXT:
                        # 핑/퐁 처리
                        if msg.data == "ping":
                            await ws.send_str("pong")
                            continue
                        elif msg.data == "pong":
                            continue
                        
                        # JSON 메시지 처리
                        try:
                            data = json.loads(msg.data)
                            if data.get("type") == "image":
                                # 이미지 데이터 처리
                                image_data = data.get("image_data")
                                if image_data:
                                    # 이미지 디코딩 및 처리
                                    loop = get_event_loop()
                                    future = loop.run_in_executor(
                                        thread_pool, 
                                        process_image_in_thread, 
                                        image_data
                                    )
                                    image = await future
                                    
                                    if image is not None:
                                        # 타임스탬프 파싱
                                        try:
                                            timestamp = datetime.fromisoformat(data.get("timestamp"))
                                        except (ValueError, TypeError):
                                            timestamp = datetime.now()
                                        
                                        # 동기화 버퍼에 저장
                                        frame_data = {
                                            "image": image,
                                            "time": timestamp
                                        }
                                        
                                        if camera_id in sync_buffer:
                                            sync_buffer[camera_id].append(frame_data)
                                        
                                        # 이미지 큐에도 저장
                                        if camera_id not in image_queues:
                                            image_queues[camera_id] = queue.Queue(maxsize=1)
                                        
                                        if not image_queues[camera_id].full():
                                            image_queues[camera_id].put(frame_data)
                        except json.JSONDecodeError:
                            # JSON 오류 무시
                            pass
                except asyncio.TimeoutError:
                    # 타임아웃은 정상이므로 무시
                    pass
    except asyncio.TimeoutError:
        # 연결 타임아웃
        update_connection_status(camera_id, "disconnected")
    except aiohttp.ClientConnectorError:
        # 서버 연결 실패
        update_connection_status(camera_id, "disconnected")
    except Exception:
        # 기타 예외
        update_connection_status(camera_id, "disconnected")
    
    # 함수 종료시 연결 해제 상태로 설정
    update_connection_status(camera_id, "disconnected")
    
    # 지수 백오프로 재연결 지연 계산 (최대 30초)
    backoff_delay = min(30, RECONNECT_DELAY * (2 ** (connection_attempts[camera_id] - 1)))
    await asyncio.sleep(backoff_delay)
    
    return False

# 메인 UI
def main():
    global selected_cameras, is_running
    
    st.title("KOMI 웹캠 모니터링")
    # st.caption(f"서버 연결: {API_URL} (시간 오프셋: {server_time_offset:.3f}초)")
    
    # 서버 상태 확인
    if st.session_state.server_status is None:
        with st.spinner("서버 연결 중..."):
            st.session_state.cameras = get_cameras()
    
    # 서버 상태에 따른 처리
    if st.session_state.server_status == "연결 실패":
        st.error("서버에 연결할 수 없습니다")
        if st.button("재연결"):
            st.session_state.cameras = get_cameras()
            st.rerun()
        return
    
    # 카메라 목록이 없으면 표시 후 종료
    if not st.session_state.cameras:
        st.info("연결된 카메라가 없습니다")
        if st.button("새로고침"):
            st.session_state.cameras = get_cameras()
            st.rerun()
        return
    
    # 카메라 다중 선택기
    if st.session_state.cameras:
        # 기본값 설정 (이전에 선택된 카메라 유지)
        default_cameras = st.session_state.selected_cameras if st.session_state.selected_cameras else st.session_state.cameras[:min(2, len(st.session_state.cameras))]
        
        selected = st.multiselect(
            "모니터링할 카메라 선택 (최대 2대)",
            st.session_state.cameras,
            default=default_cameras,
            max_selections=2
        )
        
        if selected != st.session_state.selected_cameras:
            st.session_state.selected_cameras = selected
            selected_cameras = selected
            # 카메라 선택 변경 시 동기화 버퍼 초기화
            init_sync_buffer(selected)
    
    # 동기화 설정
    col1, col2 = st.columns([3, 1])
    with col1:
        # 비어있는 텍스트 표시 (동기화 상태 준비 중 메시지 제거)
        st.caption("")
    with col2:
        # 체크박스 제거하고 동기화는 항상 활성화
        use_sync = True
    
    # 두 개의 열로 이미지 배치
    if st.session_state.selected_cameras:
        cols = st.columns(min(2, len(st.session_state.selected_cameras)))
        image_slots = {}
        status_slots = {}
        connection_indicators = {}
        
        # 각 카메라별 이미지 슬롯 생성
        for i, camera_id in enumerate(st.session_state.selected_cameras[:2]):
            with cols[i]:
                header_col1, header_col2 = st.columns([4, 1])
                with header_col1:
                    st.subheader(f"카메라 {i+1}: {camera_id}")
                with header_col2:
                    # 연결 상태 표시
                    connection_indicators[camera_id] = st.empty()
                
                image_slots[camera_id] = st.empty()
                status_slots[camera_id] = st.empty()
                status_slots[camera_id].text("실시간 스트리밍 준비 중...")
    
    # 별도 스레드 시작 (단 한번만)
    if 'thread_started' not in st.session_state:
        thread = threading.Thread(target=run_async_loop, daemon=True)
        thread.start()
        st.session_state.thread_started = True
    
    # 메인 UI 업데이트 루프
    try:
        update_interval = 0
        while True:
            update_interval += 1
            update_ui = False
            
            if use_sync and len(st.session_state.selected_cameras) > 1:
                # 동기화된 프레임 찾기
                sync_frames = find_synchronized_frames()
                if sync_frames:
                    # 동기화된 프레임이 있으면 UI 업데이트
                    for camera_id, frame_data in sync_frames.items():
                        st.session_state.camera_images[camera_id] = frame_data["image"]
                        st.session_state.image_update_time[camera_id] = frame_data["time"]
                    update_ui = True
            else:
                # 동기화 없이 각 카메라의 최신 프레임 사용
                for camera_id in st.session_state.selected_cameras[:2]:
                    if camera_id in image_queues and not image_queues[camera_id].empty():
                        try:
                            img_data = image_queues[camera_id].get(block=False)
                            st.session_state.camera_images[camera_id] = img_data.get("image")
                            st.session_state.image_update_time[camera_id] = img_data.get("time")
                            update_ui = True
                        except queue.Empty:
                            pass
            
            # 이미지 업데이트
            if update_ui:
                for camera_id in st.session_state.selected_cameras[:2]:
                    if camera_id in st.session_state.camera_images:
                        img = st.session_state.camera_images[camera_id]
                        if img is not None:
                            image_slots[camera_id].image(img, use_container_width=True)
                            status_time = st.session_state.image_update_time[camera_id].strftime('%H:%M:%S.%f')[:-3]
                            status_slots[camera_id].text(f"업데이트: {status_time}")
            
            # UI 업데이트 간격 (더 빠른 응답성)
            time.sleep(0.05)
    except Exception as e:
        # 오류 표시 개선
        st.error(f"오류가 발생했습니다. 페이지를 새로 고침해주세요.")
    
# 애플리케이션 시작
if __name__ == "__main__":
    try:
        main()
    except Exception:
        st.error("앱 실행 오류가 발생했습니다. 페이지를 새로 고침해주세요.")
    finally:
        # 종료 플래그 설정
        is_running = False
        time.sleep(0.5)
        
        # 스레드 풀 종료
        thread_pool.shutdown() 