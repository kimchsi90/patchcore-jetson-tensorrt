# PatchCore의 Jetson Xavier NX TensorRT 이식

이상 탐지 모델 PatchCore를 Jetson Xavier NX(GPU와 DLA 2개)에서 TensorRT로 실행하고, RK3588 NPU 연구([patchcore-rk3588-npu](https://github.com/kimchsi90/patchcore-rk3588-npu))와 같은 항목을 재현하는 저장소임. NPU 설계를 그대로 옮기지 않고, Jetson의 특성에 맞게 최적화와 측정 방법을 바꾸는 것이 목표임.

상태: 진행 중. JetPack 5.1.4(TensorRT 8.5.2)에서 15개 카테고리 정확도, 속도, 전력, 열 안정성을 측정함. DLA 정확도와 그래프 수준 변환의 TensorRT 속도는 측정 전임. 자세한 내용은 [docs/REPORT_KR.md](docs/REPORT_KR.md)에 있음.

## 현재까지의 결과

MVTec AD, 입력 256, JetPack 5.1.4임.

| 구성 (bottle) | pixel AUROC | AUPRO | FPS (최고 성능 모드) | 장당 에너지 |
|:---:|:---:|:---:|:---:|:---:|
| fp32 onnxruntime (PC 기준) | 0.9848 | 0.9430 | 해당 없음 | 해당 없음 |
| TensorRT fp16 | 0.9848 | 0.9429 | 116 | 130 mJ |
| int8 전체, 엔트로피 보정 | 0.7628 | 0.2796 | 해당 없음 | 해당 없음 |
| int8 전체, minmax 보정 | 0.9833 | 0.9341 | 217 | 70 mJ |
| stem과 layer1.0만 fp16, 나머지 int8 | 0.9849 | 0.9430 | 189 | 80 mJ |

- 앞쪽 층만 fp16으로 두는 혼합 정밀도가 fp32와 같은 정확도로 fp16보다 처리량이 60% 이상 높음 (10W 기본 모드 100 FPS 대 62 FPS, 최고 성능 모드 189 FPS 대 116 FPS).
- 15개 카테고리 평균에서 혼합 정밀도의 image AUROC는 0.9802(fp32 0.9816), pixel AUROC는 0.9784(0.9780), AUPRO는 0.9218(0.9215)임.
- DLA 코어 2개와 GPU를 동시에 쓰면 GPU 단독보다 처리량이 37~41% 늘고 장당 에너지는 오히려 낮음(DLA int8 엔진의 정확도는 미측정).
- 10분 연속 실행에서 GPU 클럭이 1109 MHz로 유지되고 온도는 69.5°C에서 멈춤.
- anomalib 기본 설정 baseline은 TensorRT fp16에서 64 ms이며 RK3588의 977 ms보다 빠름. 정확도는 image 1.0000, pixel 0.9854임.
- screw에서 coreset 1%를 5%로 올리면 pixel AUROC 0.9801에서 0.9889, AUPRO 0.9127에서 0.9409가 됨.
- 후처리와 원본 해상도 리사이즈를 ONNX 그래프 안으로 옮겨도 정확도가 유지됨(PC 검증).

## 폴더 구성

| 폴더 | 내용 |
|:---:|:---:|
| `trt/` | ONNX 전처리 추가, onnxruntime과 TensorRT 덤프, 점수 계산, ctypes 기반 TensorRT 실행기 |
| `trt/pc/` | PC에서 데이터 준비, ONNX 생성, 기준 정확도, 구성 탐색을 돌리는 PowerShell 스크립트 |
| `board/` | 보드에서 쓰는 성능 모드, 팬 제어, 속도 측정 스위트, QSPI 쓰기 스크립트 |
| `tools/` | SD 카드 이미지 쓰기와 검증 스크립트 |
| `results/` | 카테고리별 기준 정확도, 구성 탐색, 속도 측정 요약과 개별 결과 JSON |
| `docs/` | 보고서와 포트폴리오 문서 |

## 재현 방법

1. [patchcore-rk3588-npu](https://github.com/kimchsi90/patchcore-rk3588-npu)를 복제하고 `tools/configure.ps1`로 경로를 설정함.
2. MVTec AD는 공식 페이지의 카테고리별 링크에서 받음. 라이선스는 CC BY-NC-SA 4.0이며 이 저장소에는 데이터를 포함하지 않음.
3. PC에서 `trt/pc/p1_build_onnx.ps1`로 ONNX를 만들고 `trt/add_preproc.py`로 NHWC 입력과 정규화를 추가함.
4. 보드에서 `trt/trt_runner.py build`로 엔진을 만들고 `trt/dump_trt.py --backend trt`로 출력을 저장함.
5. 저장한 출력은 PC에서 `trt/score_dump.py`로 채점함.
6. 속도와 전력은 `board/speed_suite.sh`를 실행하고 `trt/parse_suite.py`로 표를 만듦.

스크립트의 경로는 `D:\jetson_test` 기준이며 환경에 맞게 바꿔야 함.

## 주의 사항

- 엔진과 보정 캐시는 TensorRT 버전과 기기에 종속되므로 JetPack을 바꾸면 다시 만들어야 함.
- QSPI 펌웨어를 쓰는 중 전원이 끊기면 보드가 부팅되지 않을 수 있음. `board/flash_qspi.sh`는 기존 QSPI 백업을 받은 뒤 사용함.
- 현재 라이선스 파일은 없음.
