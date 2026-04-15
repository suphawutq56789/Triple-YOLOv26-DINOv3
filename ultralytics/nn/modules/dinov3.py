"""
DINOv3 integration module for YOLOv26 Triple Input.

This module provides DINOv3 backbone integration for enhanced feature extraction
in civil engineering applications. The DINOv3 features are used as a pre-backbone
feature extractor before the standard YOLOv26 processing pipeline.

Based on: https://github.com/facebookresearch/dinov3
HuggingFace models: facebook/dinov3-small, facebook/dinov3-base, facebook/dinov3-large
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import warnings
from typing import Optional, Dict, Any, Tuple, List

try:
    from transformers import AutoModel, AutoImageProcessor
    from huggingface_hub import login, HfApi
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("transformers not available. Install with: pip install transformers huggingface_hub")

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    warnings.warn("timm not available. Install with: pip install timm")


def setup_huggingface_auth():
    """
    Setup HuggingFace authentication for DINOv3 model access.
    
    Returns:
        tuple: (is_authenticated, token_source)
    """
    import os
    
    # Check for HuggingFace token in various locations
    token = None
    token_source = None
    
    # Method 1: Environment variable
    if 'HUGGINGFACE_HUB_TOKEN' in os.environ:
        token = os.environ['HUGGINGFACE_HUB_TOKEN']
        token_source = "environment variable"
    elif 'HF_TOKEN' in os.environ:
        token = os.environ['HF_TOKEN']
        token_source = "environment variable (HF_TOKEN)"
    
    # Method 2: Check for existing token file
    if not token and TRANSFORMERS_AVAILABLE:
        try:
            from huggingface_hub import HfFolder
            token = HfFolder.get_token()
            if token:
                token_source = "saved token file"
        except:
            pass
    
    # Method 3: Try to login if token is provided
    if token and TRANSFORMERS_AVAILABLE:
        try:
            login(token=token, add_to_git_credential=False)
            print(f"✓ HuggingFace authentication successful (source: {token_source})")
            return True, token_source
        except Exception as e:
            print(f"⚠️ HuggingFace authentication failed: {e}")
            return False, token_source
    
    # No authentication found
    if not token:
        print("⚠️ No HuggingFace token found. DINOv3 models may not be accessible.")
        print("Please set up authentication:")
        print("  1. Get token from: https://huggingface.co/settings/tokens")
        print("  2. Set environment variable: export HUGGINGFACE_HUB_TOKEN='your_token'")
        print("  3. Or run: huggingface-cli login")
        return False, "not found"
    
    return False, "unknown"


def get_huggingface_token():
    """
    Get HuggingFace token from environment or saved location.
    
    Returns:
        str or None: The HuggingFace token if found
    """
    import os
    
    # Try environment variables first
    token = os.environ.get('HUGGINGFACE_HUB_TOKEN') or os.environ.get('HF_TOKEN')
    
    if not token and TRANSFORMERS_AVAILABLE:
        try:
            from huggingface_hub import HfFolder
            token = HfFolder.get_token()
        except:
            pass
    
    return token


class DINOv3Backbone(nn.Module):
    """
    DINOv3 backbone for feature extraction before YOLOv26 processing.
    
    This module integrates DINOv3 as a frozen feature extractor that processes
    input images and outputs features compatible with YOLOv26's architecture.
    """
    
    def __init__(
        self,
        model_name: str = "facebook/dinov3-small",
        input_channels: int = 3,
        output_channels: int = 64,
        freeze: bool = True,
        use_cls_token: bool = False,
        patch_size: int = 14,
        image_size: int = 224,
        pretrained: bool = True
    ):
        """
        Initialize DINOv3 backbone.
        
        Args:
            model_name: HuggingFace model name or local path
            input_channels: Number of input channels (3 for RGB, 9 for triple input)
            output_channels: Number of output channels for YOLOv26 compatibility
            freeze: Whether to freeze DINOv3 parameters
            use_cls_token: Whether to use classification token
            patch_size: Patch size for vision transformer
            image_size: Input image size
            pretrained: Whether to use pretrained weights
        """
        super().__init__()
        
        self.model_name = model_name
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.freeze = freeze
        self.use_cls_token = use_cls_token
        self.patch_size = patch_size
        self.image_size = image_size
        self.pretrained = pretrained
        
        # Initialize DINOv3 model
        self.dino_model = None
        self.processor = None
        self._load_model()
        
        # Feature dimension from DINOv3
        self.feature_dim = self._get_feature_dim()
        
        # Adaptation layers
        self._build_adaptation_layers()
        
        # Handle non-RGB input channels
        if input_channels != 3:
            self._adapt_input_channels()
        
        # Freeze DINOv3 if requested
        if self.freeze:
            self._freeze_backbone()
    
    def _load_model(self):
        """Load DINOv3 model from HuggingFace or local path."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers is required for DINOv3. Install with: pip install transformers huggingface_hub")
        
        # Setup HuggingFace authentication
        auth_success, auth_source = setup_huggingface_auth()
        
        try:
            print(f"Loading DINOv3 model: {self.model_name}")
            
            # Get token for authenticated requests
            token = get_huggingface_token()
            
            # Load model and processor with authentication
            self.dino_model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                token=token  # Add token for authentication
            )
            
            self.processor = AutoImageProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                token=token  # Add token for authentication
            )
            
            print(f"✓ Successfully loaded DINOv3 model: {self.model_name}")
            
        except Exception as e:
            print(f"Failed to load from HuggingFace: {e}")
            if "authentication" in str(e).lower() or "token" in str(e).lower():
                print("❌ Authentication error detected. Please check your HuggingFace token:")
                print("  1. Get token from: https://huggingface.co/settings/tokens")
                print("  2. Set environment variable: export HUGGINGFACE_HUB_TOKEN='your_token'")
                print("  3. Or run: huggingface-cli login")
            print("Trying timm fallback...")
            self._load_timm_fallback()
    
    def _load_timm_fallback(self):
        """Fallback to timm implementation if HuggingFace fails."""
        if not TIMM_AVAILABLE:
            raise ImportError("timm is required for DINOv3 fallback. Install with: pip install timm")
        
        try:
            # Map HuggingFace names to timm names (use DINOv2 as fallback)
            timm_name_map = {
                "facebook/dinov3-small": "vit_small_patch14_dinov2.lvd142m",
                "facebook/dinov3-base": "vit_base_patch14_dinov2.lvd142m",
                "facebook/dinov3-large": "vit_large_patch14_dinov2.lvd142m",
                "facebook/dinov3-vits16-pretrain-lvd1689m": "vit_small_patch14_dinov2.lvd142m",
                "facebook/dinov3-vitb16-pretrain-lvd1689m": "vit_base_patch14_dinov2.lvd142m",
                "facebook/dinov3-vitl16-pretrain-lvd1689m": "vit_large_patch14_dinov2.lvd142m"
            }

            timm_name = timm_name_map.get(self.model_name, "vit_small_patch14_dinov2")
            
            self.dino_model = timm.create_model(
                timm_name,
                pretrained=self.pretrained,
                num_classes=0,   # Remove classification head
                global_pool="",  # Remove global pooling
            )

            # Sync image_size to what timm actually loaded (may differ from requested)
            pe = getattr(self.dino_model, "patch_embed", None)
            if pe is not None and hasattr(pe, "img_size"):
                actual = pe.img_size
                self.image_size = actual[0] if isinstance(actual, (list, tuple)) else int(actual)

            print(f"✓ DINOv3FPN loaded (timm): {timm_name} @ img_size={self.image_size}")
            
        except Exception as e:
            print(f"Failed to load from timm: {e}")
            raise RuntimeError("Could not load DINOv3 from either HuggingFace or timm")
    
    def _get_feature_dim(self) -> int:
        """Get feature dimension from DINOv3 model."""
        if hasattr(self.dino_model, 'config'):
            # HuggingFace model
            return self.dino_model.config.hidden_size
        elif hasattr(self.dino_model, 'embed_dim'):
            # timm model
            return self.dino_model.embed_dim
        else:
            # Default dimensions for different model sizes
            size_map = {
                "small": 384,
                "base": 768,
                "large": 1024
            }
            for size, dim in size_map.items():
                if size in self.model_name.lower():
                    return dim
            return 384  # Default to small
    
    def _build_adaptation_layers(self):
        """Build layers to adapt DINOv3 features for YOLOv26."""
        # Calculate number of patches
        num_patches = (self.image_size // self.patch_size) ** 2
        
        # Feature adaptation layers
        self.feature_adapter = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.output_channels * 4),
            nn.GELU(),
            nn.Linear(self.output_channels * 4, self.output_channels),
            nn.ReLU(inplace=True)
        )
        
        # Spatial reshape for compatibility with conv layers
        # Reshape from [B, N, C] to [B, C, H, W]
        self.spatial_size = int(num_patches ** 0.5)
        
        # Optional convolution for spatial processing
        self.spatial_conv = nn.Conv2d(
            self.output_channels, 
            self.output_channels, 
            kernel_size=3, 
            padding=1, 
            bias=False
        )
        self.spatial_bn = nn.BatchNorm2d(self.output_channels)
        self.spatial_act = nn.ReLU(inplace=True)
    
    def _adapt_input_channels(self):
        """Adapt DINOv3 for non-RGB input (e.g., 9-channel triple input)."""
        if self.input_channels == 3:
            return
        
        print(f"Adapting DINOv3 for {self.input_channels}-channel input")
        
        # Create input adaptation layer
        self.input_adapter = nn.Conv2d(
            self.input_channels, 
            3, 
            kernel_size=1, 
            bias=False
        )
        
        # Initialize to preserve RGB information if input contains RGB channels
        with torch.no_grad():
            if self.input_channels >= 3:
                # Initialize first 3 channels as identity
                self.input_adapter.weight[:, :3, 0, 0] = torch.eye(3)
                
                # Initialize additional channels with small random weights
                if self.input_channels > 3:
                    nn.init.normal_(self.input_adapter.weight[:, 3:, 0, 0], std=0.02)
            else:
                # For input_channels < 3, use normal initialization
                nn.init.normal_(self.input_adapter.weight, std=0.02)
    
    def _freeze_backbone(self):
        """Freeze DINOv3 backbone parameters."""
        print("Freezing DINOv3 backbone parameters")
        for param in self.dino_model.parameters():
            param.requires_grad = False
        
        # Ensure batch norm layers are in eval mode during training
        def set_bn_eval(module):
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                module.eval()
        
        self.dino_model.apply(set_bn_eval)
        self.dino_model.eval()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through DINOv3 backbone.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Feature tensor [B, output_channels, H', W'] compatible with YOLOv26
        """
        B, C, H, W = x.shape
        
        # Adapt input channels if necessary
        if hasattr(self, 'input_adapter'):
            # Handle mismatch between expected channels and actual input
            if C != self.input_channels:
                if C == 3 and self.input_channels == 9:
                    # Validation/warmup with 3 channels, but model expects 9 channels
                    # Replicate the 3 channels to create 9 channels (3x RGB repetition)
                    x = x.repeat(1, 3, 1, 1)  # [B, 3, H, W] -> [B, 9, H, W]
                elif C == 9 and self.input_channels == 3:
                    # Use only first 3 channels if model expects 3 but receives 9
                    x = x[:, :3, :, :]  # [B, 9, H, W] -> [B, 3, H, W]
            
            x = self.input_adapter(x)
        
        # Resize to expected input size for DINOv3 if necessary
        if H != self.image_size or W != self.image_size:
            x = nn.functional.interpolate(
                x, 
                size=(self.image_size, self.image_size), 
                mode='bilinear', 
                align_corners=False
            )
        
        # Ensure DINOv3 is in eval mode if frozen
        if self.freeze:
            self.dino_model.eval()
        
        # Extract features from DINOv3
        with torch.set_grad_enabled(not self.freeze):
            if hasattr(self.dino_model, 'forward_features'):
                # timm model
                features = self.dino_model.forward_features(x)
                if features.dim() == 3:  # [B, N, C]
                    # Remove CLS token if present
                    if not self.use_cls_token and features.size(1) > self.spatial_size ** 2:
                        features = features[:, 1:, :]  # Remove CLS token
                else:  # [B, C, H, W]
                    features = features.flatten(2).transpose(1, 2)  # [B, H*W, C]
            else:
                # HuggingFace model
                outputs = self.dino_model(x, output_hidden_states=True)
                features = outputs.last_hidden_state  # [B, N, C]
                
                # Remove CLS token if present and not wanted
                # DINOv3 typically has CLS token at position 0
                if not self.use_cls_token and features.size(1) % (14*14) != 0:
                    features = features[:, 1:, :]  # Remove CLS token
        
        # Adapt features for YOLOv26
        features = self.feature_adapter(features)  # [B, N, output_channels]
        
        # Reshape to spatial format [B, C, H, W]
        # Calculate actual spatial size from feature dimensions
        N = features.size(1)  # Number of patches
        C = features.size(2)  # Feature channels after adaptation
        
        # Handle non-perfect square by trimming to the largest perfect square
        perfect_square_size = int(N ** 0.5)
        perfect_square_patches = perfect_square_size ** 2
        
        if N != perfect_square_patches:
            # Trim to the largest perfect square
            features = features[:, :perfect_square_patches, :]
            
        features = features.transpose(1, 2).view(
            B, self.output_channels, perfect_square_size, perfect_square_size
        )
        
        # Apply spatial processing
        features = self.spatial_conv(features)
        features = self.spatial_bn(features)
        features = self.spatial_act(features)
        
        # Resize to original spatial dimensions if needed
        if H != self.image_size or W != self.image_size:
            target_h = H // 2  # Assuming we want to downsample by 2x
            target_w = W // 2
            features = nn.functional.interpolate(
                features,
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=False
            )
        
        return features
    
    def train(self, mode: bool = True):
        """Set training mode, keeping DINOv3 frozen if specified."""
        super().train(mode)
        
        if self.freeze:
            # Keep DINOv3 in eval mode
            self.dino_model.eval()
            
            # Keep batch norm layers in eval mode
            def set_bn_eval(module):
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                    module.eval()
            
            self.dino_model.apply(set_bn_eval)
        
        return self
    
    def unfreeze_backbone(self):
        """Unfreeze DINOv3 backbone for fine-tuning."""
        print("Unfreezing DINOv3 backbone parameters")
        self.freeze = False
        for param in self.dino_model.parameters():
            param.requires_grad = True
    
    def get_feature_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get intermediate feature maps for analysis.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Dictionary of feature maps at different stages
        """
        feature_maps = {}
        
        # Input adaptation
        if hasattr(self, 'input_adapter'):
            x = self.input_adapter(x)
            feature_maps['input_adapted'] = x
        
        # DINOv3 features
        features = self.forward(x)
        feature_maps['dino_output'] = features
        
        return feature_maps


class P3FeatureEnhancer(nn.Module):
    """
    Feature enhancement module for P3 integration that uses conv operations
    instead of Vision Transformer for better compatibility with conv features.
    """
    
    def __init__(self, input_channels: int, output_channels: int, reduction_ratio: int = 4):
        """
        Initialize P3 feature enhancer.
        
        Args:
            input_channels: Number of input channels from P3 stage
            output_channels: Number of output channels for YOLOv26 compatibility
            reduction_ratio: Channel reduction ratio for bottleneck
        """
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        # Channel attention mechanism
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        hidden_channels = max(input_channels // reduction_ratio, 16)
        
        self.channel_attention = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, input_channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # Spatial attention mechanism
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
        # Feature enhancement layers
        self.feature_enhance = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, 3, padding=1, groups=input_channels),  # Depthwise
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels, output_channels, 1),  # Pointwise
            nn.BatchNorm2d(output_channels)
        )
        
        # Residual connection if dimensions match
        self.use_residual = (input_channels == output_channels)
        if not self.use_residual:
            self.residual_proj = nn.Conv2d(input_channels, output_channels, 1, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with attention-based feature enhancement.
        
        Args:
            x: Input tensor [B, C, H, W] from P3 stage
            
        Returns:
            Enhanced feature tensor [B, output_channels, H, W]
        """
        identity = x
        
        # Channel attention
        ca_weight = self.global_pool(x)
        ca_weight = self.channel_attention(ca_weight)
        x = x * ca_weight
        
        # Spatial attention
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        sa_input = torch.cat([avg_pool, max_pool], dim=1)
        sa_weight = self.spatial_attention(sa_input)
        x = x * sa_weight
        
        # Feature enhancement
        x = self.feature_enhance(x)
        
        # Residual connection
        if self.use_residual:
            x = x + identity
        else:
            identity = self.residual_proj(identity)
            x = x + identity
        
        return x


class DINOv3TripleBackbone(DINOv3Backbone):
    """
    DINOv3 backbone specifically designed for triple input processing.
    
    This variant processes the 9-channel triple input more intelligently,
    either by using three separate DINOv3 branches or a single adapted model.
    """
    
    def __init__(self, model_name: str = "facebook/dinov3-small", input_channels: int = 9, output_channels: int = 64, freeze: bool = True, use_separate_branches: bool = False, **kwargs):
        """
        Initialize DINOv3 for triple input.
        
        Args:
            model_name: HuggingFace model name or local path
            input_channels: Number of input channels (9 for triple input)
            output_channels: Number of output channels for YOLOv26 compatibility
            freeze: Whether to freeze DINOv3 parameters
            use_separate_branches: Whether to use separate DINOv3 branches for each input
            **kwargs: Arguments passed to parent class
        """
        self.use_separate_branches = use_separate_branches
        
        if use_separate_branches:
            # Override input channels for separate branches
            input_channels = 3
        
        super().__init__(
            model_name=model_name,
            input_channels=input_channels,
            output_channels=output_channels,
            freeze=freeze,
            **kwargs
        )
        
        if use_separate_branches:
            self._build_triple_branches()
    
    def _build_triple_branches(self):
        """Build separate DINOv3 branches for triple input."""
        print("Building separate DINOv3 branches for triple input")
        
        # Create additional branches (already have one from parent)
        self.dino_branch2 = type(self.dino_model)(self.dino_model.config) if hasattr(self.dino_model, 'config') else \
                           timm.create_model(self.model_name, pretrained=self.pretrained, num_classes=0, global_pool="")
        self.dino_branch3 = type(self.dino_model)(self.dino_model.config) if hasattr(self.dino_model, 'config') else \
                           timm.create_model(self.model_name, pretrained=self.pretrained, num_classes=0, global_pool="")
        
        # Copy weights from the first branch
        if self.pretrained:
            self.dino_branch2.load_state_dict(self.dino_model.state_dict())
            self.dino_branch3.load_state_dict(self.dino_model.state_dict())
        
        # Freeze additional branches if needed
        if self.freeze:
            for param in self.dino_branch2.parameters():
                param.requires_grad = False
            for param in self.dino_branch3.parameters():
                param.requires_grad = False
            
            self.dino_branch2.eval()
            self.dino_branch3.eval()
        
        # Feature fusion layer
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.feature_dim * 3, self.feature_dim * 2),
            nn.GELU(),
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.LayerNorm(self.feature_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for triple input.
        
        Args:
            x: Input tensor [B, 9, H, W] (triple input)
            
        Returns:
            Feature tensor [B, output_channels, H', W']
        """
        if not self.use_separate_branches:
            # Use single adapted branch
            return super().forward(x)
        
        # Split triple input
        B, C, H, W = x.shape
        assert C == 9, f"Expected 9 channels for triple input, got {C}"
        
        x1 = x[:, 0:3, :, :]  # First image
        x2 = x[:, 3:6, :, :]  # Second image
        x3 = x[:, 6:9, :, :]  # Third image
        
        # Process each branch
        features1 = self._extract_dino_features(self.dino_model, x1)
        features2 = self._extract_dino_features(self.dino_branch2, x2)
        features3 = self._extract_dino_features(self.dino_branch3, x3)
        
        # Fuse features
        features_cat = torch.cat([features1, features2, features3], dim=-1)  # [B, N, 3*C]
        features_fused = self.feature_fusion(features_cat)  # [B, N, C]
        
        # Adapt and reshape features
        features = self.feature_adapter(features_fused)
        # Calculate actual spatial size from feature dimensions
        N = features.size(1)  # Number of patches
        
        # Handle non-perfect square by trimming to the largest perfect square
        perfect_square_size = int(N ** 0.5)
        perfect_square_patches = perfect_square_size ** 2
        
        if N != perfect_square_patches:
            # Trim to the largest perfect square
            features = features[:, :perfect_square_patches, :]
            
        features = features.transpose(1, 2).view(
            B, self.output_channels, perfect_square_size, perfect_square_size
        )
        
        # Apply spatial processing
        features = self.spatial_conv(features)
        features = self.spatial_bn(features)
        features = self.spatial_act(features)
        
        return features
    
    def _extract_dino_features(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Extract features from a DINOv3 branch."""
        # Resize input if necessary
        if x.size(-1) != self.image_size or x.size(-2) != self.image_size:
            x = nn.functional.interpolate(
                x, 
                size=(self.image_size, self.image_size), 
                mode='bilinear', 
                align_corners=False
            )
        
        # Extract features
        with torch.set_grad_enabled(not self.freeze):
            if hasattr(model, 'forward_features'):
                # timm model
                features = model.forward_features(x)
                if features.dim() == 3:  # [B, N, C]
                    if not self.use_cls_token and features.size(1) % (14*14) != 0:
                        features = features[:, 1:, :]  # Remove CLS token
                else:  # [B, C, H, W]
                    features = features.flatten(2).transpose(1, 2)  # [B, H*W, C]
            else:
                # HuggingFace model
                outputs = model(x, output_hidden_states=True)
                features = outputs.last_hidden_state  # [B, N, C]
                
                if not self.use_cls_token and features.size(1) % (14*14) != 0:
                    features = features[:, 1:, :]  # Remove CLS token
        
        return features


def create_dinov3_backbone(
    model_size: str = "small",
    input_channels: int = 3,
    output_channels: int = 64,
    freeze: bool = True,
    use_triple_branches: bool = False,
    **kwargs
) -> DINOv3Backbone:
    """
    Factory function to create DINOv3 backbone.
    
    Args:
        model_size: Size of DINOv3 model (small, base, large)
        input_channels: Number of input channels
        output_channels: Number of output channels
        freeze: Whether to freeze backbone
        use_triple_branches: Whether to use separate branches for triple input
        **kwargs: Additional arguments
        
    Returns:
        DINOv3Backbone instance
    """
    # Model mapping for correct HuggingFace repository names
    model_configs = {
        "small": "facebook/dinov3-vits16-pretrain-lvd1689m",
        "small_plus": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
        "base": "facebook/dinov3-vitb16-pretrain-lvd1689m", 
        "large": "facebook/dinov3-vitl16-pretrain-lvd1689m",
        "huge": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
        "giant": "facebook/dinov3-vit7b16-pretrain-lvd1689m",
        "sat_large": "facebook/dinov3-vitl16-pretrain-lvd1689m",  # Use standard large for now
        "sat_giant": "facebook/dinov3-vit7b16-pretrain-sat493m",
    }
    
    model_name = model_configs.get(model_size, model_configs["small"])
    
    if input_channels == 9 and use_triple_branches:
        return DINOv3TripleBackbone(
            model_name=model_name,
            input_channels=input_channels,
            output_channels=output_channels,
            freeze=freeze,
            use_separate_branches=True,
            **kwargs
        )
    else:
        backbone_class = DINOv3TripleBackbone if input_channels == 9 else DINOv3Backbone
        return backbone_class(
            model_name=model_name,
            input_channels=input_channels,
            output_channels=output_channels,
            freeze=freeze,
            **kwargs
        )


class DINOv3ChannelAdapter(nn.Module):
    """
    Adapter layer to bridge between fixed DINOv3 channels and scaled YOLOv26 channels.
    
    This solves the fundamental scaling conflict by keeping DINOv3 fixed while 
    adapting its output to match YOLOv26's variant-scaled channel requirements.
    """
    
    def __init__(
        self, 
        dinov3_channels: int, 
        target_channels: int,
        use_residual: bool = True,
        activation: str = "ReLU"
    ):
        """
        Initialize channel adapter.
        
        Args:
            dinov3_channels: Fixed DINOv3 output channels
            target_channels: Target YOLOv26 channels (after variant scaling)
            use_residual: Whether to use residual connection when possible
            activation: Activation function (ReLU, GELU, SiLU)
        """
        super().__init__()
        
        self.dinov3_channels = dinov3_channels
        self.target_channels = target_channels
        self.use_residual = use_residual and (dinov3_channels == target_channels)
        
        # Build adapter layers
        if dinov3_channels == target_channels and not use_residual:
            # No adaptation needed - pass through
            self.adapter = nn.Identity()
        else:
            # Channel adaptation needed
            layers = []
            
            # Main adaptation layer
            layers.append(nn.Conv2d(dinov3_channels, target_channels, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(target_channels))
            
            # Activation function
            if activation == "ReLU":
                layers.append(nn.ReLU(inplace=True))
            elif activation == "GELU":
                layers.append(nn.GELU())
            elif activation == "SiLU":
                layers.append(nn.SiLU(inplace=True))
            
            self.adapter = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize adapter weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through adapter.
        
        Args:
            x: Input tensor [B, dinov3_channels, H, W]
            
        Returns:
            Output tensor [B, target_channels, H, W]
        """
        if self.use_residual:
            # Residual connection when channels match
            return x + self.adapter(x)
        else:
            # Direct adaptation
            return self.adapter(x)
    
    def __repr__(self):
        return f"DINOv3ChannelAdapter({self.dinov3_channels} → {self.target_channels})"


class DINOv3BackboneWithAdapter(nn.Module):
    """
    DINOv3 backbone with built-in channel adapter for YOLOv26 variant scaling.
    
    This combines DINOv3Backbone with DINOv3ChannelAdapter to create a single
    module that handles both feature extraction and channel adaptation.
    """
    
    def __init__(
        self,
        model_name: str = "facebook/dinov3-small",
        input_channels: int = 3,
        dinov3_output_channels: int = 64,  # Fixed DINOv3 output
        target_channels: int = 64,  # Target YOLOv26 channels (will be scaled)
        freeze: bool = True,
        **kwargs
    ):
        """
        Initialize DINOv3 with adapter.
        
        Args:
            model_name: HuggingFace model name
            input_channels: Number of input channels
            dinov3_output_channels: Fixed DINOv3 output channels (not scaled)
            target_channels: Target YOLOv26 channels (will be scaled by variant)
            freeze: Whether to freeze DINOv3
            **kwargs: Additional arguments for DINOv3Backbone
        """
        super().__init__()
        
        # Fixed DINOv3 backbone (never scaled)
        self.dinov3 = DINOv3Backbone(
            model_name=model_name,
            input_channels=input_channels,
            output_channels=dinov3_output_channels,  # Fixed output
            freeze=freeze,
            **kwargs
        )
        
        # Channel adapter (handles scaling)
        self.adapter = DINOv3ChannelAdapter(
            dinov3_channels=dinov3_output_channels,
            target_channels=target_channels
        )
        
        self.dinov3_output_channels = dinov3_output_channels
        self.target_channels = target_channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through DINOv3 and adapter.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Adapted features [B, target_channels, H', W']
        """
        # Extract fixed DINOv3 features
        dinov3_features = self.dinov3(x)
        
        # Adapt channels to YOLOv26 requirements
        adapted_features = self.adapter(dinov3_features)
        
        return adapted_features
    
    def update_target_channels(self, new_target_channels: int):
        """
        Update target channels for different YOLOv26 variants.
        
        Args:
            new_target_channels: New target channel count
        """
        if new_target_channels != self.target_channels:
            print(f"Updating adapter: {self.target_channels} → {new_target_channels} channels")
            
            # Create new adapter
            self.adapter = DINOv3ChannelAdapter(
                dinov3_channels=self.dinov3_output_channels,
                target_channels=new_target_channels
            )
            self.target_channels = new_target_channels
    
    def train(self, mode: bool = True):
        """Set training mode, keeping DINOv3 frozen if specified."""
        super().train(mode)
        self.dinov3.train(mode)
        return self
    
    def __repr__(self):
        return f"DINOv3BackboneWithAdapter(DINOv3: {self.dinov3_output_channels}, Target: {self.target_channels})"


# ---------------------------------------------------------------------------
# YOLOv26-GPR: Multi-scale DINOv3 Cross-Attention Architecture
# ---------------------------------------------------------------------------

class DINOv3FPN(nn.Module):
    """
    DINOv3 Feature Pyramid pre-extractor for YOLOv26-GPR.

    Runs DINOv3 once at the start of the backbone, caches intermediate
    features at three FPN scales (P3 / P4 / P5) so downstream
    DINOv3CrossFusion layers can retrieve them without re-running ViT.

    This module is a **passthrough**: it returns the raw input tensor
    unchanged, so the CNN backbone sees the full 9-ch (or 3-ch) image.
    Channel tracking in parse_model therefore treats c2 = c1.

    YAML usage (backbone layer 0)::

        - [-1, 1, DINOv3FPN, ["facebook/dinov2-small", true, 224]]
          # args: [model_name, freeze, image_size]
    """

    _active_instance = None  # class-level pointer; updated every forward pass

    # Map model-name keywords → ViT hidden dim
    _DIM_MAP = {"large": 1024, "base": 768, "small": 384}

    def __init__(
        self,
        model_name: str = "facebook/dinov2-small",
        input_channels: int = 3,
        freeze: bool = True,
        image_size: int = 224,
    ):
        super().__init__()
        self.model_name = model_name
        self.input_channels = input_channels
        self.freeze = freeze
        self.image_size = image_size
        # DINOv3 uses patch16, DINOv2 uses patch14
        self.patch_size = 16 if "dinov3" in model_name.lower() or "vits16" in model_name.lower() or "vitb16" in model_name.lower() or "vitl16" in model_name.lower() else 14

        # ---- load DINO model ------------------------------------------------
        self.use_timm = False
        self.dino_model = self._load_dino()

        # ---- feature dimension ----------------------------------------------
        self.feature_dim = self._get_dim()

        # ---- input channel adapter (e.g. 9-ch GPR → 3-ch) ------------------
        if input_channels != 3:
            self.input_adapter = nn.Conv2d(input_channels, 3, 1, bias=False)
            with torch.no_grad():
                nn.init.zeros_(self.input_adapter.weight)
                k = min(input_channels, 3)
                self.input_adapter.weight[:k, :k, 0, 0] = torch.eye(k)
        else:
            self.input_adapter = None

        # ---- freeze ViT weights --------------------------------------------
        if freeze:
            for p in self.dino_model.parameters():
                p.requires_grad = False

        # ---- runtime cache (filled during forward) -------------------------
        self._cached: dict = {}          # {'p3': tensor, 'p4': tensor, 'p5': tensor}
        self._cache_key: tuple = ()      # (B, H, W) – invalidate when input changes

    # ------------------------------------------------------------------
    def _load_dino(self) -> nn.Module:
        if TRANSFORMERS_AVAILABLE:
            try:
                setup_huggingface_auth()
                token = get_huggingface_token()
                m = AutoModel.from_pretrained(
                    self.model_name, trust_remote_code=True, token=token
                )
                print(f"✓ DINOv3FPN loaded (HuggingFace): {self.model_name}")
                return m
            except Exception as e:
                print(f"DINOv3FPN: HuggingFace failed ({e}), trying timm…")

        if TIMM_AVAILABLE:
            _map = {
                # DINOv2 (patch14)
                "facebook/dinov2-small":  "vit_small_patch14_dinov2.lvd142m",
                "facebook/dinov2-base":   "vit_base_patch14_dinov2.lvd142m",
                "facebook/dinov2-large":  "vit_large_patch14_dinov2.lvd142m",
                # DINOv3 (patch16) — fall back to DINOv2 weights via timm
                "facebook/dinov3-small":                    "vit_small_patch14_dinov2.lvd142m",
                "facebook/dinov3-base":                     "vit_base_patch14_dinov2.lvd142m",
                "facebook/dinov3-vits16-pretrain-lvd1689m": "vit_small_patch14_dinov2.lvd142m",
                "facebook/dinov3-vitb16-pretrain-lvd1689m": "vit_base_patch14_dinov2.lvd142m",
                "facebook/dinov3-vitl16-pretrain-lvd1689m": "vit_large_patch14_dinov2.lvd142m",
                "facebook/dinov3-vitg16-pretrain-lvd1689m": "vit_giant_patch14_dinov2.lvd142m",
                "facebook/dinov3-vitl16-pretrain-sat493m":  "vit_large_patch14_dinov2.lvd142m",
                "facebook/dinov3-vitg16-pretrain-sat493m":  "vit_giant_patch14_dinov2.lvd142m",
            }
            name = _map.get(self.model_name, "vit_small_patch14_dinov2.lvd142m")
            m = timm.create_model(name, pretrained=True, num_classes=0, global_pool="",
                                  dynamic_img_size=True)
            self.use_timm = True
            # Sync self.image_size to what timm actually uses (e.g. 518 for dinov2)
            pe = getattr(m, "patch_embed", None)
            if pe is not None:
                actual = getattr(pe, "img_size", None)
                if actual is not None:
                    self.image_size = actual[0] if isinstance(actual, (list, tuple)) else int(actual)
            print(f"✓ DINOv3FPN loaded (timm): {name} @ img_size={self.image_size}")
            return m

        raise RuntimeError(
            "DINOv3FPN: install 'transformers' or 'timm'.\n"
            "  pip install transformers huggingface_hub  OR  pip install timm"
        )

    def _get_dim(self) -> int:
        if hasattr(self.dino_model, "config"):
            return self.dino_model.config.hidden_size
        if hasattr(self.dino_model, "embed_dim"):
            return self.dino_model.embed_dim
        # Check by model name keyword (covers both dinov2 and dinov3 naming)
        n = self.model_name.lower()
        if "vitl" in n or "large" in n:
            return 1024
        if "vitb" in n or "base" in n:
            return 768
        if "vitg" in n or "giant" in n:
            return 1536
        return 384  # vits / small default

    # ------------------------------------------------------------------
    def _extract(self, x: torch.Tensor) -> dict:
        """Run ViT and return {'p3', 'p4', 'p5'} spatial feature maps."""
        B = x.shape[0]
        h = w = self.image_size // self.patch_size   # e.g. 224/14 = 16

        with torch.set_grad_enabled(not self.freeze):
            if not self.use_timm:
                # HuggingFace path: output_hidden_states=True
                out = self.dino_model(x, output_hidden_states=True)
                hs = out.hidden_states   # tuple([B, 1+N, D]) including CLS
                n = len(hs)
                layers = {
                    "p3": hs[max(1, n // 3)][:, 1:, :],          # early
                    "p4": hs[max(1, 2 * n // 3)][:, 1:, :],      # mid
                    "p5": hs[-1][:, 1:, :],                       # final
                }
            else:
                # timm path
                if hasattr(self.dino_model, "get_intermediate_layers"):
                    nb = len(self.dino_model.blocks)
                    idxs = [nb // 3 - 1, 2 * nb // 3 - 1, nb - 1]
                    feats = self.dino_model.get_intermediate_layers(x, n=idxs)
                    layers = {"p3": feats[0], "p4": feats[1], "p5": feats[2]}
                else:
                    f = self.dino_model.forward_features(x)
                    if f.dim() == 3:
                        f = f[:, 1:, :]    # remove CLS
                    layers = {"p3": f, "p4": f, "p5": f}

        D = self.feature_dim

        def to_spatial(t):
            # t: [B, N, D]  →  [B, D, sq, sq]
            n = t.shape[1]
            sq = int(n ** 0.5)
            t = t[:, : sq * sq, :]
            return t.transpose(1, 2).reshape(B, D, sq, sq)

        return {k: to_spatial(v) for k, v in layers.items()}

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Cache DINOv3 multi-scale features; return input unchanged."""
        B, C, H, W = x.shape
        key = (B, H, W)

        # Adapt input channels for ViT
        if self.input_adapter is not None:
            dino_x = self.input_adapter(x)
        elif C != 3:
            dino_x = x[:, :3]
        else:
            dino_x = x

        # Resize to ViT input size
        if H != self.image_size or W != self.image_size:
            dino_x = F.interpolate(
                dino_x, size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False
            )

        if self.freeze:
            self.dino_model.eval()

        self._cached = self._extract(dino_x)
        self._cache_key = key
        DINOv3FPN._active_instance = self   # publish for DINOv3CrossFusion

        return x   # passthrough

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze and self.dino_model is not None:
            self.dino_model.eval()
        return self

    def __repr__(self):
        return (
            f"DINOv3FPN(model={self.model_name!r}, "
            f"dim={self.feature_dim}, freeze={self.freeze})"
        )


# ---------------------------------------------------------------------------

class DINOv3CrossFusion(nn.Module):
    """
    Cross-attention fusion between CNN features and DINOv3 FPN features.

    Retrieves cached features from ``DINOv3FPN._active_instance`` and
    enhances CNN features via::

        out = CNN_feat + γ · CrossAttn(Q=CNN, K=V=DINOv3)

    The learnable scale ``γ`` (``nn.Parameter``) is initialised to 0,
    meaning at the start of training the module behaves like an identity;
    it gradually learns to blend ViT knowledge into the CNN stream.

    Must be preceded in the backbone by a ``DINOv3FPN`` layer.
    Falls back to identity if DINOv3FPN has not run yet (e.g. debugging).

    YAML usage::

        - [-1, 1, DINOv3CrossFusion, ["p3", 4]]
          # args: [scale, num_heads]   (channels prepended by tasks.py)
    """

    def __init__(
        self,
        channels: int,
        scale: str = "p3",
        num_heads: int = 4,
        dino_dim: int = 384,        # ViT-S=384, ViT-B=768, ViT-L=1024
    ):
        super().__init__()
        self.channels = channels
        self.scale = scale
        self.num_heads = num_heads
        self.dino_dim = dino_dim

        # Project DINOv3 features to CNN channel space
        self.dino_proj = nn.Sequential(
            nn.Conv2d(dino_dim, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

        # Normalisation before attention
        self.norm_q = nn.LayerNorm(channels)
        self.norm_k = nn.LayerNorm(channels)

        # Multi-head cross-attention (Q from CNN, K/V from DINOv3)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=max(1, num_heads),
            batch_first=True,
            dropout=0.0,
        )

        # Output conv + residual gate (γ=0 at init → identity at start)
        self.out_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.out_proj.weight)
        for seq in [self.dino_proj]:
            for m in seq:
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out")

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: CNN feature map  [B, C, H, W]
        Returns:
            DINOv3-enhanced feature map  [B, C, H, W]
        """
        B, C, H, W = x.shape

        fpn = DINOv3FPN._active_instance
        if fpn is None or self.scale not in fpn._cached:
            return x   # safety fallback

        dino_feat = fpn._cached[self.scale].to(x.device)   # [B, D, h, w]

        # Project DINOv3 dim → CNN channels
        dino_feat = self.dino_proj(dino_feat)               # [B, C, h, w]

        # Resize DINOv3 spatial map to match CNN feature size
        if dino_feat.shape[-2:] != (H, W):
            dino_feat = F.interpolate(
                dino_feat, size=(H, W), mode="bilinear", align_corners=False
            )

        # ---- Cross-attention: Q from CNN, K/V from DINOv3 ----------------
        # Q: [B, H*W, C]   K,V: [B, H*W, C]
        q  = x.flatten(2).transpose(1, 2)           # [B, N_q, C]
        kv = dino_feat.flatten(2).transpose(1, 2)   # [B, N_kv, C]

        q  = self.norm_q(q)
        kv = self.norm_k(kv)

        attn_out, _ = self.cross_attn(q, kv, kv)    # [B, N_q, C]
        attn_out = attn_out.transpose(1, 2).reshape(B, C, H, W)
        attn_out = self.out_proj(attn_out)

        return x + self.gamma * attn_out

    def __repr__(self):
        return (
            f"DINOv3CrossFusion(ch={self.channels}, scale={self.scale!r}, "
            f"heads={self.num_heads}, dino_dim={self.dino_dim})"
        )


# ---------------------------------------------------------------------------
# YOLOv26-GPR: MedSAM ViT-B Feature Pyramid (Medical Domain Backbone)
# ---------------------------------------------------------------------------

class MedSAMFPN(nn.Module):
    """
    MedSAM ViT-B feature pyramid pre-extractor for YOLOv26-GPR.

    Loads MedSAM image encoder (ViT-B, trained on 1.5M medical images
    including ultrasound — closest public domain to GPR).  Runs ViT once,
    caches P3/P4/P5 features via forward hooks, then returns the original
    input unchanged (passthrough) so the CNN backbone sees the full image.

    Domain rationale:
        Ultrasound ≈ GPR: both use pulse-echo wave physics, produce
        hyperbolic diffraction patterns, and contain layered reflections.
        MedSAM features therefore transfer far better than DINOv3 natural-
        image features.

    YAML usage (backbone layer 0)::

        - [-1, 1, MedSAMFPN, ["wanglab/medsam-vit-base", true, 512]]
          # args: [checkpoint, freeze, image_size]
          # image_size 512 = good speed/quality trade-off (MedSAM native=1024)
    """

    _active_instance = None   # class-level pointer; updated every forward
    FEATURE_DIM = 768         # ViT-B hidden dim (fixed)
    PATCH_SIZE  = 16          # SAM uses patch16

    def __init__(
        self,
        checkpoint: str = "wanglab/medsam-vit-base",
        input_channels: int = 3,
        freeze: bool = True,
        image_size: int = 512,
    ):
        super().__init__()
        self.checkpoint    = checkpoint
        self.input_channels = input_channels
        self.freeze        = freeze
        self.image_size    = image_size
        self.feature_dim   = self.FEATURE_DIM
        self.patch_size    = self.PATCH_SIZE
        self.use_timm      = False

        # ---- input channel adapter (9-ch GPR → 3-ch for ViT) ---------------
        if input_channels != 3:
            self.input_adapter = nn.Conv2d(input_channels, 3, 1, bias=False)
            with torch.no_grad():
                nn.init.zeros_(self.input_adapter.weight)
                k = min(input_channels, 3)
                self.input_adapter.weight[:k, :k, 0, 0] = torch.eye(k)
        else:
            self.input_adapter = None

        # ---- load MedSAM ViT-B image encoder --------------------------------
        self._hook_outputs: dict = {}
        self.vit = self._load_vit()

        # ---- register hooks at P3 / P4 / P5 ViT depths ---------------------
        self._hooks = []
        self._register_hooks()

        # ---- freeze ---------------------------------------------------------
        if freeze:
            for p in self.vit.parameters():
                p.requires_grad = False

        # ---- runtime cache --------------------------------------------------
        self._cached: dict  = {}
        self._cache_key: tuple = ()

    # ------------------------------------------------------------------
    def _load_vit(self) -> nn.Module:
        """Load MedSAM image encoder.  Priority: HF SamModel → timm → error."""

        # --- HuggingFace path (SamModel) -------------------------------------
        if TRANSFORMERS_AVAILABLE:
            try:
                setup_huggingface_auth()
                token = get_huggingface_token()
                from transformers import SamModel
                full = SamModel.from_pretrained(
                    self.checkpoint, trust_remote_code=True, token=token
                )
                vit = full.vision_encoder   # SamVisionEncoder
                # SamVisionEncoder enforces exact 1024×1024 — resize pos_embed
                # so we can use a smaller image_size (e.g. 512) without error
                self._resize_pos_embed_hf(vit, self.image_size)
                print(f"✓ MedSAMFPN loaded (HuggingFace): {self.checkpoint}")
                return vit
            except Exception as e:
                print(f"MedSAMFPN: HuggingFace failed ({e}), trying timm…")

        # --- timm path (samvit_base_patch16) ---------------------------------
        # NOTE: timm loads SAM-pretrained weights (not MedSAM fine-tune)
        # but the architecture is identical — acceptable fallback
        if TIMM_AVAILABLE:
            try:
                m = timm.create_model(
                    "samvit_base_patch16",
                    pretrained=True,
                    num_classes=0,
                    global_pool="",
                )
                self.use_timm = True
                print("✓ MedSAMFPN loaded (timm): samvit_base_patch16")
                return m
            except Exception as e:
                print(f"MedSAMFPN: timm failed ({e})")

        raise RuntimeError(
            "MedSAMFPN: cannot load MedSAM. Install dependencies:\n"
            "  pip install transformers huggingface_hub   (recommended)\n"
            "  OR  pip install timm>=0.9"
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _resize_pos_embed_hf(vit, target_size: int):
        """
        Interpolate SamVisionEncoder absolute pos_embed to support
        a custom input size (default MedSAM native = 1024).

        pos_embed shape: [1, H_grid, W_grid, D]  where H_grid = native // patch_size
        After this call vit.image_size is patched so the size-check passes.
        """
        native = 1024
        patch  = 16
        if target_size == native:
            return

        if not hasattr(vit, "pos_embed") or vit.pos_embed is None:
            # Just patch the size check — no learnable pos_embed to resize
            vit.image_size = (target_size, target_size)
            return

        orig_grid = native    // patch   # 64
        tgt_grid  = target_size // patch   # e.g. 32 for 512

        pe = vit.pos_embed.data                   # [1, orig_grid, orig_grid, D]
        pe = pe.permute(0, 3, 1, 2)               # [1, D, orig_grid, orig_grid]
        pe = F.interpolate(
            pe.float(),
            size=(tgt_grid, tgt_grid),
            mode="bicubic",
            align_corners=False,
        )
        pe = pe.permute(0, 2, 3, 1).contiguous()  # [1, tgt_grid, tgt_grid, D]
        vit.pos_embed = nn.Parameter(pe)

        # Patch all image_size attributes so the internal size checks pass
        vit.image_size = (target_size, target_size)
        if hasattr(vit, "patch_embed") and hasattr(vit.patch_embed, "image_size"):
            vit.patch_embed.image_size = (target_size, target_size)
        # Patch any nested image_size inside attention layers
        for module in vit.modules():
            if hasattr(module, "image_size") and module is not vit:
                try:
                    module.image_size = (target_size, target_size)
                except Exception:
                    pass
        print(f"  MedSAMFPN: pos_embed resized {orig_grid}×{orig_grid} → {tgt_grid}×{tgt_grid}")

    # ------------------------------------------------------------------
    def _register_hooks(self):
        """Attach forward hooks at 1/3, 2/3, and final ViT blocks."""

        def _make_hook(name: str):
            def hook(module, input, output):
                # HF SamVisionEncoderLayer  → output is (hidden_states, ...)
                # timm ViT block             → output is hidden_states tensor
                t = output[0] if isinstance(output, (tuple, list)) else output
                self._hook_outputs[name] = t
            return hook

        if self.use_timm:
            blocks = self.vit.blocks
        else:
            # HuggingFace SamVisionEncoder exposes .layers
            blocks = self.vit.layers

        nb = len(blocks)
        indices = {
            "p3": max(0, nb // 3 - 1),
            "p4": max(0, 2 * nb // 3 - 1),
            "p5": nb - 1,
        }
        for scale, idx in indices.items():
            h = blocks[idx].register_forward_hook(_make_hook(scale))
            self._hooks.append(h)

    def __getstate__(self):
        """Remove unpicklable hooks before serialization."""
        for h in self._hooks:
            h.remove()
        state = self.__dict__.copy()
        state["_hooks"] = []
        return state

    def __setstate__(self, state):
        """Restore state and re-register hooks after deserialization."""
        self.__dict__.update(state)
        self._hooks = []
        self._register_hooks()

    # ------------------------------------------------------------------
    @staticmethod
    def _to_spatial(t: torch.Tensor, B: int) -> torch.Tensor:
        """
        Convert ViT block output to [B, C, H, W].

        SAM/MedSAM window-attention layers emit [B, H, W, C].
        timm ViT blocks emit [B, N, C] (flattened tokens).
        """
        if t.dim() == 4:          # [B, H, W, C]  — SAM spatial format
            return t.permute(0, 3, 1, 2).contiguous()
        elif t.dim() == 3:        # [B, N, C]  — standard ViT token format
            N, C = t.shape[1], t.shape[2]
            sq = int(N ** 0.5)
            return t[:, :sq * sq, :].transpose(1, 2).reshape(B, C, sq, sq)
        return t

    # ------------------------------------------------------------------
    def _extract(self, x: torch.Tensor) -> dict:
        """Run ViT (triggers hooks) and convert cached outputs to spatial maps."""
        self._hook_outputs.clear()
        B = x.shape[0]

        with torch.set_grad_enabled(not self.freeze):
            if self.use_timm:
                self.vit.forward_features(x)
            else:
                self.vit(x)   # HF SamVisionEncoder forward

        return {
            k: self._to_spatial(v, B)
            for k, v in self._hook_outputs.items()
        }

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Cache MedSAM multi-scale features; return input unchanged."""
        B, C, H, W = x.shape

        # Adapt 9-ch input for ViT
        if self.input_adapter is not None:
            vit_x = self.input_adapter(x)
        elif C != 3:
            vit_x = x[:, :3]
        else:
            vit_x = x

        # Resize to ViT input size
        if H != self.image_size or W != self.image_size:
            vit_x = F.interpolate(
                vit_x, size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False
            )

        if self.freeze:
            self.vit.eval()

        self._cached   = self._extract(vit_x)
        self._cache_key = (B, H, W)
        MedSAMFPN._active_instance = self   # publish for MedSAMCrossFusion

        return x   # passthrough

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze and self.vit is not None:
            self.vit.eval()
        return self

    def __repr__(self):
        return (
            f"MedSAMFPN(ckpt={self.checkpoint!r}, "
            f"dim={self.feature_dim}, freeze={self.freeze}, "
            f"img_size={self.image_size})"
        )


# ---------------------------------------------------------------------------

class MedSAMCrossFusion(nn.Module):
    """
    Cross-attention fusion: CNN features (Q) × MedSAM features (K, V).

    Drop-in replacement for DINOv3CrossFusion.  Reads cached features from
    ``MedSAMFPN._active_instance`` instead of ``DINOv3FPN``.

    out = CNN_feat + γ · CrossAttn(Q=CNN, K=V=MedSAM)

    γ is initialised to 0 (identity at training start), so the module is
    safe even if MedSAM features are not yet useful.

    YAML usage::

        - [-1, 1, MedSAMCrossFusion, ["p3", 4]]
          # args: [scale, num_heads]
    """

    def __init__(
        self,
        channels: int,
        scale: str = "p3",
        num_heads: int = 4,
        medsam_dim: int = 768,   # ViT-B = 768 (fixed for MedSAM)
    ):
        super().__init__()
        self.channels   = channels
        self.scale      = scale
        self.num_heads  = num_heads
        self.medsam_dim = medsam_dim

        # Project MedSAM dim (768) → CNN channel space
        self.medsam_proj = nn.Sequential(
            nn.Conv2d(medsam_dim, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

        self.norm_q = nn.LayerNorm(channels)
        self.norm_k = nn.LayerNorm(channels)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=max(1, num_heads),
            batch_first=True,
            dropout=0.0,
        )

        self.out_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.gamma    = nn.Parameter(torch.zeros(1))   # γ=0 init

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.out_proj.weight)
        for m in self.medsam_proj:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        fpn = MedSAMFPN._active_instance
        if fpn is None or self.scale not in fpn._cached:
            return x   # safety fallback

        med_feat = fpn._cached[self.scale].to(x.device)   # [B, D, h, w]

        # Project 768 → CNN channels
        med_feat = self.medsam_proj(med_feat)              # [B, C, h, w]

        # Resize MedSAM spatial map → CNN feature size
        if med_feat.shape[-2:] != (H, W):
            med_feat = F.interpolate(
                med_feat, size=(H, W), mode="bilinear", align_corners=False
            )

        # Cross-attention: Q=CNN, K=V=MedSAM
        q  = x.flatten(2).transpose(1, 2)          # [B, N, C]
        kv = med_feat.flatten(2).transpose(1, 2)   # [B, N, C]

        q  = self.norm_q(q)
        kv = self.norm_k(kv)

        attn_out, _ = self.cross_attn(q, kv, kv)
        attn_out = attn_out.transpose(1, 2).reshape(B, C, H, W)
        attn_out = self.out_proj(attn_out)

        return x + self.gamma * attn_out

    def __repr__(self):
        return (
            f"MedSAMCrossFusion(ch={self.channels}, scale={self.scale!r}, "
            f"heads={self.num_heads}, medsam_dim={self.medsam_dim})"
        )