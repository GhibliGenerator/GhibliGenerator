import torch
import torchvision
from src.models.style_network import GhibliStyleNet
from src.temporal.temporal_coherence import TemporalCoherenceModule
from src.interpolation.frame_interpolator import AdaptiveFrameInterpolator
from src.utils.preprocessor import ImagePreprocessor
from src.utils.postprocessor import VideoPostprocessor
import argparse
import logging
from typing import Tuple, List
import numpy as np

class GhibliGenerator:
    def __init__(self, config: dict):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.style_net = GhibliStyleNet(config["model_params"]).to(self.device)
        self.temporal_module = TemporalCoherenceModule(config["temporal_params"])
        self.frame_interpolator = AdaptiveFrameInterpolator(config["interp_params"])
        self.preprocessor = ImagePreprocessor()
        self.postprocessor = VideoPostprocessor()
        
        self._load_pretrained_weights()
        self._setup_logging()
        
    def _load_pretrained_weights(self) -> None:
        """Load pretrained Ghibli-specific weights"""
        try:
            self.style_net.load_state_dict(torch.load("weights/ghibli_style.pth"))
        except Exception as e:
            logging.error(f"Weight loading failed: {str(e)}")
            raise
            
    def _setup_logging(self) -> None:
        """Configure logging system"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def process_image_to_video(self, input_path: str, output_path: str, 
                             duration: float = 10.0) -> None:
        """Main processing pipeline"""
        # Preprocess input
        input_tensor = self.preprocessor.process(input_path)
        
        # Generate style-transferred frames
        styled_frames = self.style_net.generate(input_tensor)
        
        # Apply temporal coherence
        coherent_frames = self.temporal_module.enforce_coherence(styled_frames)
        
        # Interpolate frames
        final_frames = self.frame_interpolator.interpolate(coherent_frames, 
                                                         target_fps=24,
                                                         duration=duration)
        
        # Post-process and save video
        self.postprocessor.save_video(final_frames, output_path)
        
def main():
    parser = argparse.ArgumentParser(description="Ghibli Generator CLI")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output", required=True, help="Output video path")
    parser.add_argument("--duration", type=float, default=10.0, help="Video duration")
    
    args = parser.parse_args()
    
    config = {
        "model_params": {"layers": 32, "channels": 64},
        "temporal_params": {"window_size": 5, "consistency_weight": 0.8},
        "interp_params": {"method": "adaptive", "complexity": 4}
    }
    
    generator = GhibliGenerator(config)
    generator.process_image_to_video(args.input, args.output, args.duration)

if __name__ == "__main__":
    main()
