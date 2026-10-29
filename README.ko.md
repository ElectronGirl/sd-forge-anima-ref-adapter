<div align="center">

<img src="https://raw.githubusercontent.com/ElectronGirl/ComfyUI-Anima-Ref-Adapter/main/docs/showcase/banner.jpg" width="100%" alt="Anima Reference Adapter v2">

# Anima Reference Adapter for Forge Neo

**레퍼런스 이미지에서 캐릭터와 그림체를, 프롬프트에서 자세·장소·구도를.**

[![Weights](https://img.shields.io/badge/%F0%9F%A4%97%20weights-anima--ref--adapter-ffcc4d)](https://huggingface.co/ElectronGirl/anima-ref-adapter)
[![ComfyUI](https://img.shields.io/badge/%F0%9F%A7%A9%20ComfyUI-nodes-1f6feb)](https://github.com/ElectronGirl/ComfyUI-Anima-Ref-Adapter)
[![Training code](https://img.shields.io/badge/%F0%9F%94%AC%20training-code-8a63d2)](https://github.com/ElectronGirl/anima-ref-adapter-research)
[![Ko-fi](https://img.shields.io/badge/Ko--fi-support%20training-ff5e5b?logo=ko-fi&logoColor=white)](https://ko-fi.com/electrongirl)

[English](README.md) · **한국어** · [日本語](README.ja.md)

</div>

<br>

[Stable Diffusion WebUI Forge - Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)에서
[Anima Reference Adapter v2](https://huggingface.co/ElectronGirl/anima-ref-adapter)를 쓰게 해 주는 확장입니다.
[Anima](https://huggingface.co/circlestone-labs/Anima)에 레퍼런스 이미지 한 장과 장면만 적은 프롬프트를 주면, 캐릭터(머리,
눈, 의상, 장신구)와 그림체의 상당 부분은 이미지에서, 자세·배경·구도는 프롬프트에서 가져옵니다. 레퍼런스를 하나 더
넣으면 두 캐릭터를 한 장에 그릴 수 있습니다.

이 확장으로 Forge Neo에서 생성한 결과입니다(RTX 3060 Ti):

<img src="docs/examples.jpg" width="100%" alt="Forge Neo results">

예시 이미지, 측정 결과, 작동 원리는
[ComfyUI 버전 README](https://github.com/ElectronGirl/ComfyUI-Anima-Ref-Adapter/blob/main/README.ko.md)에 있습니다. 이 확장은
같은 어댑터를 같은 가이던스 방식으로 실행합니다.

> 자금 부족으로 어댑터 개발은 중단되었습니다. 추가 학습을 후원하려면 **[Ko-fi](https://ko-fi.com/electrongirl)**를 이용해 주세요.

## 설치

1. Forge Neo에서 **Extensions → Install from URL**을 열고
   `https://github.com/ElectronGirl/sd-forge-anima-ref-adapter`를 붙여 넣은 뒤 **Install**을 누릅니다.
   또는 `extensions/`에 직접 받습니다:
   ```bash
   cd sd-webui-forge-neo/extensions
   git clone https://github.com/ElectronGirl/sd-forge-anima-ref-adapter.git
   ```
2. Forge Neo를 재시작합니다.
3. 체크포인트로 Anima base v1.0을 고르고, **VAE / Text Encoder**에서 `qwen_3_06b_base`와 `qwen_image_vae`를
   선택합니다. 확장 없이 Anima를 쓸 때와 같습니다.

어댑터 가중치(`anima_ref_adapter_v2_r64.safetensors`, 183MB)는 확장을 켜고 처음 생성할 때 LoRA 폴더로 자동으로
받아집니다.

## 사용법

1. txt2img에서 **Anima Reference Adapter** 패널을 열고 켭니다.
2. **Reference**에 레퍼런스 이미지를 넣습니다.
3. 프롬프트에는 퀄리티 태그, `1girl`이나 `1boy`, `solo`, 그리고 장면만 씁니다:
   `masterpiece, best quality, score_7, safe, 1girl, solo, cowboy shot, sitting, cafe, holding cup, smile, looking at viewer, indoors, window, day`
   캐릭터 이름, 작가 태그, 의상 태그는 넣지 않습니다.
4. 생성합니다. Anima의 평소 설정(ER SDE나 Euler, 30 steps, CFG 4.5)을 그대로 쓰면 됩니다.

**두 캐릭터:** **Second character (right, optional)**에 두 번째 캐릭터를 넣으면 첫 번째 이미지의 오른쪽에 붙습니다.
`solo` 대신 `2girls`(또는 `1boy, 1girl`)를 쓰고, 왼쪽과 오른쪽이 누구인지 몇 단어로 적고, 1216x832처럼 가로가 긴
크기를 씁니다.

| 설정 | 기본값 | |
|---|---|---|
| Reference guidance | 4.5 | 레퍼런스 쪽으로 미는 세기입니다. 1이면 추가 모델 호출이 없어 빠르지만 약합니다. |
| Correction | `apg` | 가이던스가 높을 때 색이 과해지는 것을 막습니다. `rescale`, `none`도 있습니다. |
| Reference guidance end | 1.0 | 샘플링 진행 중 이 지점에서 추가 가이던스를 멈춥니다. 0.6 정도면 거의 같은 그림을 더 빨리 얻습니다. |
| Reference size (MP) | 1.0 | 레퍼런스를 이 면적으로 맞춥니다. VRAM이 부족하면 낮춥니다. |

설정값은 이미지의 infotext(`anima_ref_*`)에 저장됩니다.

**turbo LoRA:** 프롬프트에 Anima base용 turbo LoRA를 넣고(예: `<lora:anima-turbo-lora-v0.2:1>`) 10 steps, CFG 1,
Reference guidance 1로 설정합니다.

## 메모리

캐시해 둔 레퍼런스는 1MP 기준 VRAM 약 1GB를 차지하며, 확장이 이만큼을 Forge의 메모리 관리자에 미리 예약합니다.
다른 프로그램과 함께 쓰는 8GB GPU에서 생성이 매우 느려지면(콘솔에 남은 메모리가 적다는 경고가 나옵니다) Forge를
`--reserve-vram 2` 옵션으로 실행하거나 레퍼런스 크기를 낮춰 주세요.

## API

항상 켜져 있는 스크립트(always-on script)이므로 `/sdapi/v1/txt2img`로도 쓸 수 있습니다:

```json
"alwayson_scripts": {
  "anima reference adapter": {
    "args": [true, "<reference as base64>", "", "anima_ref_adapter_v2_r64.safetensors", 4.5, "apg", 1.0, 1.0]
  }
}
```

인자는 순서대로 켜기 여부, 레퍼런스(base64), 두 번째 레퍼런스(base64 또는 `""`), 어댑터 파일, Reference guidance,
Correction, Reference guidance end, 레퍼런스 크기(MP)입니다.

## 참고

- Forge Neo `neo-2.31`(PyTorch 2.14, CUDA 13.0), RTX 3060 Ti 8GB의 txt2img에서 테스트했습니다. img2img와 hires fix는
  테스트하지 않았습니다.
- v2 어댑터만 지원합니다. 파일럿 어댑터는 ComfyUI 노드가 필요합니다.
- 체크포인트가 Anima여야 합니다. 다른 모델에서는 패널이 아무 동작도 하지 않습니다.

## 라이선스

확장 코드는 **Apache-2.0**([LICENSE](LICENSE))입니다. 어댑터 가중치는 CircleStone Model의 파생물이며 Hugging Face에서
**CircleStone Non-Commercial License**로 배포합니다. 이 프로젝트는 CircleStone Labs나 Forge Neo 개발자와 관계가 없고,
그들의 보증을 받지 않았습니다.
