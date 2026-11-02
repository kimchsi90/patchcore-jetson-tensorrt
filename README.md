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
- DLA 코어 2개와 GPU를 동시에 쓰면 GPU 단독(187 FPS)보다 처리량이 44% 늘어 269.5 FPS가 되고 장당 에너지는 15% 낮음(66.9 mJ). DLA int8 혼합 엔진은 15개 카테고리에서 정확도를 확인함(평균 AUPRO 차이 +0.0002). DLA 2개만으로도 140 FPS가 나와 GPU를 비워 둘 수 있음.
- 입력 리사이즈는 INTER_AREA를 표준으로 정함(15개 카테고리 평균이 Pillow 방식과 같음). INTER_LINEAR는 평균 AUPRO가 0.0047 낮음. 끝단 처리량을 가장 크게 올림. GPU 워커 하나가 61.9에서 152.6 FPS가 되고, DLA 2개만으로도 GPU 없이 128.6 FPS(장당 75 mJ)가 나옴.
- anomalib 기본 설정 baseline은 TensorRT 8.5.2 fp16에서 63.6 ms(10W 모드), 최고 성능 모드에서 25.8 FPS이며 RK3588의 977 ms보다 빠름. 최적화한 혼합 엔진(189 FPS)은 baseline보다 약 7배 빠르고 장당 에너지는 약 1/7임.
- screw에서 coreset 1%를 5%로 올리면 AUPRO가 0.9165에서 0.9470이 되지만 처리량은 174 FPS에서 93 FPS로 줄고 장당 에너지가 2배가 됨(TensorRT 혼합 엔진, 최고 성능 모드).
- 후처리와 원본 해상도 리사이즈를 ONNX 그래프 안으로 옮겨도 정확도가 유지됨. 끝단 측정(최고 성능 모드)에서 CPU 리사이즈가 병목이며, 리사이즈를 엔진 안으로 옮기면 CPU 사용이 10% 아래로 내려가고 124 FPS(GPU 워커 2개)가 나옴.

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

## LLM 단계 (같은 보드)

PatchCore 다음 단계로 같은 Xavier NX에서 언어 모델을 측정함. 상세 결과와 재현 스크립트는 [llm/README_llm.md](llm/README_llm.md)에 있음.

- llama.cpp(CUDA, sm_72)로 Qwen2.5-1.5B Q4_K_M을 실행함. 최대 성능 모드에서 프리필 758 tok/s, 디코딩 34.2 tok/s임. 10W 모드는 459 tok/s와 18.3 tok/s임.
- `llama-server`로 서빙하면 동시 접속 4개에서 총 73.5 tok/s, 첫 토큰 지연은 0.12초 이하임.
- vLLM 0.2.2는 소스 빌드와 호환용 shim 두 개로 동작함. 참고 결과로만 두었음.
- PatchCore 2-DLA 파이프라인과 LLM을 동시에 돌리면 LLM 디코딩은 31%, PatchCore는 24% 느려짐. GPU 파이프라인과 같이 돌리면 각각 59%와 67% 느려짐.

## 주의 사항

- 엔진과 보정 캐시는 TensorRT 버전과 기기에 종속되므로 JetPack을 바꾸면 다시 만들어야 함.
- QSPI 펌웨어를 쓰는 중 전원이 끊기면 보드가 부팅되지 않을 수 있음. `board/flash_qspi.sh`는 기존 QSPI 백업을 받은 뒤 사용함.
- 현재 라이선스 파일은 없음.
