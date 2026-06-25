import time
from doa.estimator import estimate_direction

def main():
    print("실시간 방향 감지 시작 (Ctrl+C로 종료)\n")
    try:
        while True:
            result = estimate_direction(None)
            if result.angle_deg is None:
                # ReSpeaker 방향값을 못 읽음 (미연결 / pyusb 미설치 / USB 권한)
                print("\r장치 없음 — ReSpeaker 미연결·pyusb 미설치·USB 권한 확인        ",
                      end="", flush=True)
            else:
                print(f"\r{result.angle_deg:>5.0f}° → {result.direction.value}        ",
                      end="", flush=True)
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\n종료")

if __name__ == "__main__":
    main()
