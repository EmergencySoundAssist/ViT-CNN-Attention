import time
from doa.estimator import estimate_direction

def main():
    print("실시간 방향 감지 시작 (Ctrl+C로 종료)\n")
    last = None
    try:
        while True:
            result = estimate_direction(None)
            print(f"{result.angle_deg:>5.0f}° → {result.direction.value}")
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\n종료")

if __name__ == "__main__":
    main()
