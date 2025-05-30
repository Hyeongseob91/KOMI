import json
import time
import asyncio
import aiohttp
import random
import queue
import collections
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# 전역 변수들
MAX_RECONNECT_ATTEMPTS = 5
RECONNECT_DELAY = 1.0  # 초 단위
connection_attempts = {}  # 카메라ID -> 시도 횟수

# 카메라별 이미지 큐
image_queues = {}  # 카메라ID -> Queue

# WebSocket 연결 상태
ws_connection_status = {}  # 카메라ID -> 상태 ("connected", "disconnected", "reconnecting")

# 스레드별 전용 세션과 이벤트 루프
thread_local = threading.local()

# 이미지 처리 관련 변수
image_processor = None  # 이미지 처리 콜백 함수 (streamlit_app.py에서 설정)

# 연결 상태 업데이트 함수
def update_connection_status(camera_id, status):
    """카메라 연결 상태 업데이트"""
    global ws_connection_status
    
    # 전역 상태 업데이트
    ws_connection_status[camera_id] = status
    
    # 연결 시도 횟수 관리
    if status == "connected":
        connection_attempts[camera_id] = 0
    elif status == "disconnected":
        if camera_id not in connection_attempts:
            connection_attempts[camera_id] = 0

# 이미지 처리 콜백 설정
def set_image_processor(processor_callback):
    """이미지 처리 콜백 함수 설정"""
    global image_processor
    image_processor = processor_callback

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

# 서버 시간 동기화 함수 - 간소화 및 안정성 향상
async def sync_server_time(API_URL, server_time_offset, last_time_sync, TIME_SYNC_INTERVAL):
    """서버 시간과 로컬 시간의 차이를 계산"""
    
    # 이미 최근에 동기화했다면 스킵
    current_time = time.time()
    if current_time - last_time_sync < TIME_SYNC_INTERVAL:
        return server_time_offset, last_time_sync, True
    
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
                return server_time_offset, last_time_sync, False
                
            local_time_after = time.time()
            data = await response.json()
            
            server_timestamp = data.get("timestamp")
            if not server_timestamp:
                return server_time_offset, last_time_sync, False
            
            network_delay = (local_time_after - local_time_before) / 2
            local_time_avg = local_time_before + network_delay
            server_time_offset = server_timestamp - local_time_avg
            last_time_sync = time.time()
            return server_time_offset, last_time_sync, True
    except asyncio.TimeoutError:
        # 타임아웃은 조용히 처리
        return server_time_offset, last_time_sync, False
    except Exception:
        # 그 외 오류도 조용히 처리
        return server_time_offset, last_time_sync, False

# 비동기 카메라 목록 가져오기
async def async_get_cameras(API_URL):
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

# WebSocket 연결 및 이미지 스트리밍 수신 - 안정성 개선
async def connect_to_camera_stream(camera_id, API_URL, is_running, sync_buffer, thread_pool):
    """WebSocket을 통해 카메라 스트림에 연결"""
    global connection_attempts, image_queues, image_processor
    
    # 이미지 처리기가 설정되지 않은 경우 오류
    if image_processor is None:
        print("이미지 처리 콜백이 설정되지 않았습니다")
        return False
    
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
                            msg_type = data.get("type")
                            
                            # 이미지 데이터 처리 (일반 이미지 또는 포즈 데이터 포함 이미지)
                            if msg_type == "image" or msg_type == "image_with_pose":
                                # 이미지 데이터 처리
                                image_data = data.get("image_data")
                                if image_data:
                                    # 이미지 디코딩 및 처리
                                    loop = get_event_loop()
                                    future = loop.run_in_executor(
                                        thread_pool, 
                                        image_processor, 
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
                                        
                                        # 포즈 데이터가 있으면 저장
                                        if msg_type == "image_with_pose" and "pose_data" in data:
                                            frame_data["pose_data"] = data["pose_data"]
                                        
                                        if camera_id in sync_buffer:
                                            sync_buffer[camera_id].append(frame_data)
                                        
                                        # 이미지 큐에 추가
                                        if camera_id in image_queues:
                                            # 큐가 꽉 찬 경우 오래된 항목 제거
                                            if image_queues[camera_id].qsize() >= 10:
                                                try:
                                                    image_queues[camera_id].get_nowait()
                                                except queue.Empty:
                                                    pass
                                            
                                            # 새 프레임 추가 (포즈 데이터 포함)
                                            image_queues[camera_id].put(frame_data)
                            
                            # 포즈 데이터 업데이트 알림 처리
                            elif msg_type == "image_update_with_pose":
                                pass  # 필요에 따라 구현
                                
                        except json.JSONDecodeError:
                            # JSON 디코딩 오류는 무시
                            pass
                except asyncio.TimeoutError:
                    # 타임아웃은 정상, 계속 진행
                    pass
                except Exception as e:
                    # 다른 오류는 루프 탈출
                    print(f"WebSocket 수신 오류: {camera_id} - {str(e)}")
                    break
                
                # 짧은 딜레이로 CPU 사용률 감소
                await asyncio.sleep(0.01)
            
            # 루프 종료 시 연결 종료
            await ws.close()
    
    except Exception as e:
        # 연결 실패 처리
        print(f"카메라 {camera_id} 스트림 연결 오류: {str(e)}")
        update_connection_status(camera_id, "disconnected")
        return False
    
    # 연결 종료됨
    update_connection_status(camera_id, "disconnected")
    return False

# 선택된 카메라에 대한 이미지 업데이트 및 동기화 처리
async def update_images(API_URL, selected_cameras, is_running, sync_buffer, server_time_offset, 
                       last_time_sync, TIME_SYNC_INTERVAL, thread_pool):
    """선택된 카메라들의 이미지 스트림 수신 및 동기화"""
    global image_queues
    
    # 각 카메라별 큐 초기화
    for camera_id in selected_cameras:
        if camera_id not in image_queues:
            image_queues[camera_id] = queue.Queue(maxsize=20)
    
    # 동기화 버퍼 초기화
    for camera_id in selected_cameras:
        if camera_id not in sync_buffer:
            sync_buffer[camera_id] = collections.deque(maxlen=10)
    
    # 카메라별 연결 관리 태스크 시작
    camera_tasks = {}
    
    while is_running:
        # 서버 시간 동기화
        server_time_offset, last_time_sync, _ = await sync_server_time(
            API_URL, server_time_offset, last_time_sync, TIME_SYNC_INTERVAL
        )
        
        # 선택된 카메라 변경 감지 및 동기화
        for camera_id in list(camera_tasks.keys()):
            if camera_id not in selected_cameras:
                # 제거된 카메라 작업 취소
                if not camera_tasks[camera_id].done():
                    camera_tasks[camera_id].cancel()
                del camera_tasks[camera_id]
        
        # 새로운 카메라 연결 작업 시작
        for camera_id in selected_cameras:
            if camera_id not in camera_tasks or camera_tasks[camera_id].done():
                # 새 작업 생성
                camera_tasks[camera_id] = asyncio.create_task(
                    connect_to_camera_stream(camera_id, API_URL, is_running, sync_buffer, thread_pool)
                )
        
        # 카메라 작업 상태 확인 및 오류 처리
        for camera_id, task in list(camera_tasks.items()):
            if task.done():
                exception = task.exception()
                if exception:
                    print(f"카메라 {camera_id} 작업 오류: {str(exception)}")
                
                # 재연결 시도
                camera_tasks[camera_id] = asyncio.create_task(
                    connect_to_camera_stream(camera_id, API_URL, is_running, sync_buffer, thread_pool)
                )
        
        # 주기적인 작업 대기
        await asyncio.sleep(1.0)
    
    # 종료 시 모든 작업 취소
    for task in camera_tasks.values():
        if not task.done():
            task.cancel()
    
    # 모든 작업 완료 대기
    await asyncio.gather(*camera_tasks.values(), return_exceptions=True)

# 이미지 처리 비동기 스레드 실행
def run_async_loop(API_URL, selected_cameras, is_running, sync_buffer, server_time_offset, 
                  last_time_sync, TIME_SYNC_INTERVAL, process_image_callback, thread_pool):
    """비동기 이미지 처리 메인 루프 (별도 스레드에서 실행)"""
    try:
        # 이미지 처리 콜백 설정
        set_image_processor(process_image_callback)
        
        # 이벤트 루프 설정
        loop = get_event_loop()
        
        # 비동기 태스크 실행
        loop.run_until_complete(
            update_images(
                API_URL, selected_cameras, is_running, sync_buffer, 
                server_time_offset, last_time_sync, TIME_SYNC_INTERVAL, thread_pool
            )
        )
    except Exception as e:
        print(f"비동기 루프 오류: {str(e)}")
    finally:
        # 리소스 정리
        try:
            loop.run_until_complete(close_session())
        except:
            pass

