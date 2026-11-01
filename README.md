# PatchCore의 Jetson Xavier NX TensorRT 이식

이상 탐지 모델 PatchCore를 Jetson Xavier NX(GPU와 DLA 2개)에서 TensorRT로 실행하고, RK3588 NPU 연구([patchcore-rk3588-npu](https://github.com/kimchsi90/patchcore-rk3588-npu))와 같은 항목을 재현하는 저장소임. NPU 설계를 그대로 옮기지 않고, Jetson의 특성에 맞게 최적화와 측정 방법을 바꾸는 것이 목표임.

상태: 진행 중. JetPack 4.6.1(TensorRT 8.2.1)의 참고 측정값이 있고, 최신 JetPack 5.1.4(TensorRT 8.5) 측정은 보드 이미지 교체 후 진행함. 최종 수치는 [docs/REPORT_KR.md](docs/REPORT_KR.md)에 갱신함.

## 현재까지의 결과

MVTec AD bottle, 입력 256, JetPack 4.6.1, 참고값임.

| 구성 | pixel AUROC | AUPRO | GPU 연산 시간 |
|:---:|:---:|:---:|:---:|
| fp32 onnxruntime (PC 기준) | 0.9848 | 0.9430 | 해당 없음 |
| TensorRT fp16 | 0.9848 | 0.9429 | 16.1 ms |
| TensorRT int8 전체, 엔트로피 보정 | 0.7655 | 0.2850 | 9.1 ms |
| TensorRT int8 전체, minmax 보정 | 0.9832 | 0.9328 | 9.1 ms |
| stem과 layer1.0만 fp16, 나머지 int8 | 0.9849 | 0.9431 | 10.2 ms |

- 앞쪽 층만 fp16으로 두는 혼합 정밀도가 fp32와 같은 정확도로 fp16보다 37% 빠름 (10W 모드).
- 최고 성능 모드(GPU 1109 MHz)에서는 혼합 정밀도 182 FPS(5.4 ms), fp16 116 FPS(8.5 ms)임.
- anomalib 기본 설정 baseline은 TensorRT fp16에서 64 ms이며 RK3588의 977 ms보다 빠름. 정확도는 image 1.0000, pixel 0.9854임.
- screw에서 coreset 1%를 5%로 올리면 pixel AUROC 0.9801에서 0.9889, AUPRO 0.9127에서 0.9409가 됨.

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
