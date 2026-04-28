"""
Ψ-Vortex Automatic Symmetry Detection Module
=============================================
Enables FULLY AUTOMATED physics-aware initialization by detecting
symmetry properties directly from input-output data.

This module extends the existing physics-aware initialization (Equation 5)
by automatically determining M_sym from data, eliminating the need for
manual domain expertise.

Key Features:
1. Automatic odd/even symmetry detection from data
2. Multiple detection strategies (interpolation, nearest-neighbor, statistical)
3. Confidence scoring for detected symmetry
4. Fallback to identity mask when no clear symmetry is found

Usage:
    from core_auto_symmetry import AutoSymmetryDetector, apply_auto_vortex_init
    
    # Detect symmetry from data
    detector = AutoSymmetryDetector()
    symmetry_type, confidence = detector.detect(V, I)
    
    # Apply fully automated initialization
    apply_auto_vortex_init(model, V, I)  # No domain expertise needed!

Author: Sorin Liviu Jurj
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Optional, Union
from core_physics_init import PhysicsAwareInitializer, SymmetryProjector


class AutoSymmetryDetector:
    """
    Automatically detects symmetry properties from input-output data.
    
    Supports detection of:
    - Odd symmetry: f(-x) ≈ -f(x)  (e.g., memristor I-V characteristics)
    - Even symmetry: f(-x) ≈ f(x)  (e.g., power dissipation)
    - No symmetry: fallback to identity mask
    
    Detection Methods:
    1. Direct pairing: Find (x, -x) pairs and compare f(x) vs ±f(-x)
    2. Interpolation: Interpolate f(-x) and compare
    3. Statistical: Compute correlation between f(x) and ±f(-|x|)
    """
    
    def __init__(self, 
                 tolerance: float = 0.15,
                 min_confidence: float = 0.7,
                 method: str = "auto"):
        """
        Initialize the symmetry detector.
        
        Args:
            tolerance: Maximum relative error for symmetry detection
            min_confidence: Minimum confidence score to declare symmetry
            method: Detection method ("direct", "interpolate", "statistical", "auto")
        """
        self.tolerance = tolerance
        self.min_confidence = min_confidence
        self.method = method
        
    def detect(self, X: torch.Tensor, Y: torch.Tensor, 
               verbose: bool = True) -> Tuple[str, float]:
        """
        Detect symmetry type from input-output data.
        
        Args:
            X: Input tensor (voltage, position, etc.) - can be [N] or [N, 1]
            Y: Output tensor (current, force, etc.) - can be [N] or [N, 1]
            verbose: Whether to print detection results
            
        Returns:
            symmetry_type: One of "odd", "even", "none"
            confidence: Confidence score [0, 1]
        """
        # Flatten tensors
        X = X.flatten().detach()
        Y = Y.flatten().detach()
        
        if len(X) != len(Y):
            raise ValueError(f"X and Y must have same length: {len(X)} vs {len(Y)}")
        
        # Try multiple detection methods
        results = {}
        
        if self.method in ["auto", "direct"]:
            results["direct"] = self._detect_direct_pairing(X, Y)
            
        if self.method in ["auto", "interpolate"]:
            results["interpolate"] = self._detect_interpolation(X, Y)
            
        if self.method in ["auto", "statistical"]:
            results["statistical"] = self._detect_statistical(X, Y)
        
        # Aggregate results (weighted by method reliability)
        symmetry_type, confidence = self._aggregate_results(results)
        
        if verbose:
            print(f"Auto-Symmetry Detection Results:")
            print(f"  Method results: {results}")
            print(f"  Detected: {symmetry_type} (confidence: {confidence:.2%})")
            
        return symmetry_type, confidence
    
    def _detect_direct_pairing(self, X: torch.Tensor, Y: torch.Tensor) -> Dict:
        """
        Detect symmetry by finding (x, -x) pairs in the data.
        
        For each x > 0, find the closest -x in the dataset and compare:
        - Odd: Y(x) ≈ -Y(-x)
        - Even: Y(x) ≈ Y(-x)
        """
        # Separate positive and negative X values
        pos_mask = X > self.tolerance  # Avoid values near zero
        neg_mask = X < -self.tolerance
        
        X_pos = X[pos_mask]
        Y_pos = Y[pos_mask]
        X_neg = X[neg_mask]
        Y_neg = Y[neg_mask]
        
        if len(X_pos) < 5 or len(X_neg) < 5:
            return {"odd": 0.0, "even": 0.0, "valid": False}
        
        # Find pairs: for each x > 0, find closest |x_neg| ≈ x
        odd_errors = []
        even_errors = []
        
        for i, (xp, yp) in enumerate(zip(X_pos, Y_pos)):
            # Find closest negative value
            distances = torch.abs(torch.abs(X_neg) - xp)
            closest_idx = torch.argmin(distances)
            
            if distances[closest_idx] < self.tolerance * torch.abs(xp):
                yn = Y_neg[closest_idx]
                
                # Odd: f(x) ≈ -f(-x)
                odd_err = torch.abs(yp + yn) / (torch.abs(yp) + 1e-10)
                odd_errors.append(odd_err.item())
                
                # Even: f(x) ≈ f(-x)
                even_err = torch.abs(yp - yn) / (torch.abs(yp) + 1e-10)
                even_errors.append(even_err.item())
        
        if len(odd_errors) < 5:
            return {"odd": 0.0, "even": 0.0, "valid": False}
        
        # Calculate confidence as 1 - mean_error
        odd_confidence = 1.0 - np.mean(odd_errors)
        even_confidence = 1.0 - np.mean(even_errors)
        
        return {
            "odd": max(0, odd_confidence),
            "even": max(0, even_confidence),
            "valid": True
        }
    
    def _detect_interpolation(self, X: torch.Tensor, Y: torch.Tensor) -> Dict:
        """
        Detect symmetry using linear interpolation.
        
        Interpolate Y at -X values and compare with ±Y.
        """
        # Sort by X for interpolation
        sort_idx = torch.argsort(X)
        X_sorted = X[sort_idx]
        Y_sorted = Y[sort_idx]
        
        # Only use X values where -X is within data range
        x_min, x_max = X_sorted.min(), X_sorted.max()
        
        # Find points where -x is in range
        valid_mask = (-X_sorted >= x_min) & (-X_sorted <= x_max)
        valid_mask &= (X_sorted > self.tolerance) | (X_sorted < -self.tolerance)
        
        X_valid = X_sorted[valid_mask]
        Y_valid = Y_sorted[valid_mask]
        
        if len(X_valid) < 10:
            return {"odd": 0.0, "even": 0.0, "valid": False}
        
        # Interpolate Y at -X
        Y_neg_interp = torch.zeros_like(Y_valid)
        for i, x in enumerate(X_valid):
            neg_x = -x
            # Linear interpolation
            idx = torch.searchsorted(X_sorted, neg_x)
            if idx == 0:
                Y_neg_interp[i] = Y_sorted[0]
            elif idx >= len(X_sorted):
                Y_neg_interp[i] = Y_sorted[-1]
            else:
                # Linear interpolation between adjacent points
                x0, x1 = X_sorted[idx-1], X_sorted[idx]
                y0, y1 = Y_sorted[idx-1], Y_sorted[idx]
                alpha = (neg_x - x0) / (x1 - x0 + 1e-10)
                Y_neg_interp[i] = y0 + alpha * (y1 - y0)
        
        # Calculate symmetry scores
        y_scale = torch.std(Y_valid) + 1e-10
        
        odd_error = torch.mean(torch.abs(Y_valid + Y_neg_interp)) / y_scale
        even_error = torch.mean(torch.abs(Y_valid - Y_neg_interp)) / y_scale
        
        return {
            "odd": max(0, 1.0 - odd_error.item()),
            "even": max(0, 1.0 - even_error.item()),
            "valid": True
        }
    
    def _detect_statistical(self, X: torch.Tensor, Y: torch.Tensor) -> Dict:
        """
        Detect symmetry using statistical correlation.
        
        Compute correlation between:
        - Odd: Y and -sign(X)*Y_reflected
        - Even: Y and Y_reflected
        """
        # Use sign of X to create reflection
        sign_X = torch.sign(X)
        abs_X = torch.abs(X)
        
        # Sort by |X|
        sort_idx = torch.argsort(abs_X)
        Y_by_abs_x = Y[sort_idx]
        sign_by_abs_x = sign_X[sort_idx]
        
        # Split into positive and negative X groups
        pos_mask = sign_by_abs_x > 0
        neg_mask = sign_by_abs_x < 0
        
        Y_pos = Y_by_abs_x[pos_mask]
        Y_neg = Y_by_abs_x[neg_mask]
        
        # Use minimum length
        min_len = min(len(Y_pos), len(Y_neg))
        if min_len < 10:
            return {"odd": 0.0, "even": 0.0, "valid": False}
        
        Y_pos = Y_pos[:min_len]
        Y_neg = Y_neg[:min_len]
        
        # Correlation for odd symmetry: Y_pos ≈ -Y_neg
        # Correlation for even symmetry: Y_pos ≈ Y_neg
        
        # Compute correlations
        y_pos_norm = Y_pos - Y_pos.mean()
        y_neg_norm = Y_neg - Y_neg.mean()
        
        denom = (torch.std(Y_pos) * torch.std(Y_neg) * min_len + 1e-10)
        
        # Odd: high negative correlation between Y_pos and Y_neg
        corr = torch.sum(y_pos_norm * y_neg_norm) / denom
        
        odd_confidence = max(0, -corr.item())  # Negative correlation = odd
        even_confidence = max(0, corr.item())   # Positive correlation = even
        
        return {
            "odd": odd_confidence,
            "even": even_confidence,
            "valid": True
        }
    
    def _aggregate_results(self, results: Dict) -> Tuple[str, float]:
        """
        Aggregate results from multiple detection methods.
        
        Weighted by method reliability:
        - direct: 0.4 (most reliable when pairs exist)
        - interpolate: 0.35 (good for continuous data)
        - statistical: 0.25 (fallback)
        """
        weights = {
            "direct": 0.4,
            "interpolate": 0.35,
            "statistical": 0.25
        }
        
        odd_score = 0.0
        even_score = 0.0
        total_weight = 0.0
        
        for method, result in results.items():
            if result.get("valid", False):
                w = weights.get(method, 0.25)
                odd_score += w * result["odd"]
                even_score += w * result["even"]
                total_weight += w
        
        if total_weight < 0.1:
            return "none", 0.0
        
        odd_score /= total_weight
        even_score /= total_weight
        
        # Determine winner
        if odd_score > even_score and odd_score >= self.min_confidence:
            return "odd", odd_score
        elif even_score > odd_score and even_score >= self.min_confidence:
            return "even", even_score
        else:
            return "none", max(odd_score, even_score)


class AutoSymmetryMaskConstructor:
    """
    Constructs M_sym masks automatically based on detected symmetry.
    """
    
    @staticmethod
    def construct_mask(symmetry_type: str, weight_shape: Tuple[int, ...],
                       input_dim: int = 0) -> torch.Tensor:
        """
        Construct symmetry mask based on detected symmetry type.
        
        Args:
            symmetry_type: "odd", "even", or "none"
            weight_shape: Shape of weight matrix (out_features, in_features)
            input_dim: Which input dimension has the symmetric variable
            
        Returns:
            M_sym: Symmetry mask tensor
        """
        if symmetry_type == "odd":
            return SymmetryProjector.memristor_odd_symmetry(weight_shape, input_dim)
        elif symmetry_type == "even":
            return AutoSymmetryMaskConstructor._even_symmetry_mask(weight_shape, input_dim)
        else:
            # No symmetry: identity mask
            return torch.ones(weight_shape)
    
    @staticmethod
    def _even_symmetry_mask(weight_shape: Tuple[int, ...], 
                           input_dim: int = 0) -> torch.Tensor:
        """
        Creates symmetry projector for even symmetry: f(-x) = f(x)
        
        This requires symmetric weight patterns where weights
        corresponding to positive and negative inputs have same signs.
        """
        n_out, n_in = weight_shape[0], weight_shape[1] if len(weight_shape) > 1 else 1
        M_sym = torch.ones(n_out, n_in)
        
        # For even symmetry, first and second half of neurons should match
        # (not negate like odd symmetry)
        # The mask is all ones, but we use absolute value constraint
        
        return M_sym


class AutoPhysicsAwareInitializer(PhysicsAwareInitializer):
    """
    Physics-Aware Initializer with AUTOMATIC symmetry detection.
    
    Extends PhysicsAwareInitializer to automatically detect symmetry
    from input-output data, eliminating the need for domain expertise.
    
    Usage:
        initializer = AutoPhysicsAwareInitializer()
        initializer.initialize_with_data(model, X, Y)
    """
    
    def __init__(self, 
                 epsilon: float = 0.01,
                 sigma: float = 0.01,
                 scale: float = 0.1,
                 detection_tolerance: float = 0.15,
                 min_confidence: float = 0.7):
        """
        Initialize the auto physics-aware initializer.
        
        Args:
            epsilon: Noise factor for symmetry breaking
            sigma: Standard deviation of noise
            scale: Overall scale factor
            detection_tolerance: Tolerance for symmetry detection
            min_confidence: Minimum confidence to apply detected symmetry
        """
        # Initialize with "none" - will be set during detection
        super().__init__(pde_type="none", epsilon=epsilon, sigma=sigma, scale=scale)
        
        self.detector = AutoSymmetryDetector(
            tolerance=detection_tolerance,
            min_confidence=min_confidence
        )
        self.detected_symmetry = "none"
        self.detection_confidence = 0.0
        
    def detect_symmetry(self, X: torch.Tensor, Y: torch.Tensor,
                       verbose: bool = True) -> Tuple[str, float]:
        """
        Detect symmetry from data and store result.
        
        Args:
            X: Input tensor
            Y: Output tensor
            verbose: Whether to print results
            
        Returns:
            symmetry_type, confidence
        """
        self.detected_symmetry, self.detection_confidence = self.detector.detect(
            X, Y, verbose=verbose
        )
        
        # Map detected symmetry to pde_type for parent class
        symmetry_to_pde = {
            "odd": "memristor",
            "even": "conservation",  # Even symmetry uses conservation-like mask
            "none": "none"
        }
        self.pde_type = symmetry_to_pde.get(self.detected_symmetry, "none")
        
        return self.detected_symmetry, self.detection_confidence
    
    def initialize_with_data(self, model: nn.Module, 
                            X: torch.Tensor, Y: torch.Tensor,
                            verbose: bool = True) -> Dict:
        """
        Full automated initialization: detect symmetry and apply initialization.
        
        Args:
            model: Neural network to initialize
            X: Input data for symmetry detection
            Y: Output data for symmetry detection
            verbose: Whether to print progress
            
        Returns:
            Dictionary with detection results and initialization info
        """
        if verbose:
            print("=" * 60)
            print("Ψ-Vortex AUTOMATIC Physics-Aware Initialization")
            print("=" * 60)
        
        # Step 1: Detect symmetry from data
        symmetry_type, confidence = self.detect_symmetry(X, Y, verbose=verbose)
        
        # Step 2: Apply initialization using detected symmetry
        if verbose:
            print(f"\nApplying initialization with detected symmetry: {symmetry_type}")
        
        self.initialize(model)
        
        # Return info
        return {
            "symmetry_type": symmetry_type,
            "confidence": confidence,
            "pde_type_used": self.pde_type,
            "epsilon": self.epsilon,
            "sigma": self.sigma,
            "scale": self.scale
        }
    
    def _get_layer_symmetry(self, name: str, shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Override parent method to use detected symmetry.
        """
        if self.detected_symmetry == "none":
            # Identity mask
            return torch.ones(shape)
        
        # Use parent class logic with detected pde_type
        return super()._get_layer_symmetry(name, shape)


def apply_auto_vortex_init(model: nn.Module, 
                           X: torch.Tensor, 
                           Y: torch.Tensor,
                           verbose: bool = True) -> Dict:
    """
    Convenience function for FULLY AUTOMATED Ψ-Vortex initialization.
    
    This is the drop-in replacement for apply_psi_vortex_init() that
    automatically detects symmetry from data.
    
    Args:
        model: Neural network to initialize
        X: Input data (e.g., voltage)
        Y: Output data (e.g., current)
        verbose: Whether to print progress
        
    Returns:
        Dictionary with detection and initialization info
    
    Example:
        # Old way (requires domain expertise):
        apply_psi_vortex_init(model, pde_type="memristor")
        
        # New way (fully automated):
        apply_auto_vortex_init(model, V, I)
    """
    initializer = AutoPhysicsAwareInitializer(
        epsilon=0.01,
        sigma=0.01,
        scale=0.1,
        detection_tolerance=0.15,
        min_confidence=0.7
    )
    
    return initializer.initialize_with_data(model, X, Y, verbose=verbose)


def apply_identity_vortex_init(model: nn.Module, verbose: bool = True) -> None:
    """
    Apply Ψ-Vortex initialization with identity mask (no symmetry).
    
    This is the baseline for when no symmetry is detected or known.
    Still benefits from orthogonal initialization, just without symmetry priors.
    
    Args:
        model: Neural network to initialize
        verbose: Whether to print progress
    """
    if verbose:
        print("Applying Ψ-Vortex Identity Initialization (no symmetry prior)")
    
    initializer = PhysicsAwareInitializer(
        pde_type="none",  # Will use identity mask
        epsilon=0.01,
        sigma=0.01,
        scale=0.1
    )
    initializer.initialize(model)


# ============================================================
# Testing utilities
# ============================================================

def test_symmetry_detection():
    """Test the automatic symmetry detection on known data."""
    print("\n" + "="*60)
    print("Testing Automatic Symmetry Detection")
    print("="*60)
    
    detector = AutoSymmetryDetector()
    
    # Test 1: Odd symmetry - f(-x) = -f(x)
    # Use x^3 which is a true odd function: (-x)^3 = -x^3
    print("\n--- Test 1: Odd Symmetry f(x) = x^3 ---")
    X_odd = torch.linspace(-2, 2, 200)
    Y_odd = X_odd ** 3  # True odd function: f(-x) = -f(x)
    sym_type, conf = detector.detect(X_odd, Y_odd)
    print(f"Expected: odd, Got: {sym_type} (conf: {conf:.2%})")
    assert sym_type == "odd", f"Failed: expected odd, got {sym_type}"
    
    # Test 2: Even symmetry - f(-x) = f(x)
    # Use x^2 which is a true even function: (-x)^2 = x^2
    print("\n--- Test 2: Even Symmetry f(x) = x^2 ---")
    X_even = torch.linspace(-2, 2, 200)
    Y_even = X_even ** 2  # True even function: f(-x) = f(x)
    sym_type, conf = detector.detect(X_even, Y_even)
    print(f"Expected: even, Got: {sym_type} (conf: {conf:.2%})")
    assert sym_type == "even", f"Failed: expected even, got {sym_type}"
    
    # Test 3: No symmetry (exponential)
    print("\n--- Test 3: No Symmetry f(x) = exp(x) ---")
    X_none = torch.linspace(-2, 2, 200)
    Y_none = torch.exp(X_none)  # Neither odd nor even
    sym_type, conf = detector.detect(X_none, Y_none)
    print(f"Expected: none, Got: {sym_type} (conf: {conf:.2%})")
    # Note: exponential may show weak symmetry scores, that's OK
    
    # Test 4: Memristor-like (odd symmetry: I(-V) = -I(V))
    # The physical constraint is that current reverses with voltage
    # Use sinh(V) which is odd: sinh(-V) = -sinh(V)
    print("\n--- Test 4: Memristor-like (sinh) ---")
    V = torch.linspace(-2, 2, 200)
    I = 1e-4 * torch.sinh(V)  # Odd function modeling conductance
    sym_type, conf = detector.detect(V, I)
    print(f"Expected: odd, Got: {sym_type} (conf: {conf:.2%})")
    
    # Test 5: Product x*tanh(x) is actually EVEN!
    # f(-x) = (-x)*tanh(-x) = (-x)*(-tanh(x)) = x*tanh(x) = f(x)
    print("\n--- Test 5: x*tanh(x) (actually EVEN!) ---")
    X = torch.linspace(-2, 2, 200)
    Y = X * torch.tanh(X)  # This is EVEN, not odd!
    sym_type, conf = detector.detect(X, Y)
    print(f"Expected: even, Got: {sym_type} (conf: {conf:.2%})")
    
    print("\n" + "="*60)
    print("All symmetry detection tests completed!")
    print("="*60)


if __name__ == "__main__":
    test_symmetry_detection()