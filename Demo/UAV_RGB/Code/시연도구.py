#!/usr/bin/env python3
"""원본 사진에서 모델 추론 CSV·Excel·512 오버레이를 생성합니다."""
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="SAM3 원본 사진 추론·손상 산출물 생성")
    commands = parser.add_subparsers(dest="command", required=True)
    infer = commands.add_parser("infer", help="원본 사진→모델 추론→좌표·정량 CSV·512 이미지")
    infer.add_argument("--images", required=True, help="원본 JPG/PNG 한 장 또는 사진 폴더")
    infer.add_argument("--model-record", required=True, help="SAM3의 학습기록.json; 모델 검증 정보")
    infer.add_argument("--output", required=True, help="새 결과 폴더; 기존 결과 덮어쓰기 금지")
    infer.add_argument("--mesh", required=True, help="기존 EPSG:5186/Z-up OBJ")
    infer.add_argument("--asset-manifest", help="OBJ 검증 정보; 생략 시 시연 폴더 기본값")
    infer.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    infer.add_argument("--threshold", type=float, default=0.5)
    infer.add_argument("--ray-backend", choices=["auto", "warp", "trimesh"], default="auto")
    infer.add_argument("--warp-device", default="cpu", help="Ray 계산 장치")
    infer.add_argument("--save-diagnostics", action="store_true",
                       help="검증이 필요할 때만 예측 마스크·상세 계산 기록도 저장")
    args = parser.parse_args()
    from demo512.prediction_pipeline import export_predictions
    result = export_predictions(args.images, args.output, args.model_record, args.mesh,
                                args.asset_manifest, device=args.device,
                                threshold=args.threshold, ray_backend=args.ray_backend,
                                warp_device=args.warp_device,
                                save_diagnostics=args.save_diagnostics)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
