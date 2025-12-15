#########################
# Gradient Monitor
# Author: Koureas Stavros
#########################

class GradientMonitor:
    """
    Monitors gradients during training to detect vanishing or exploding gradients.

    Thresholds:
    - Vanishing: gradient norm < vanish_threshold (default: 1e-7)
    - Exploding: gradient norm > explode_threshold (default: 1e3)
    """

    def __init__(self, mp, params, vanish_threshold=1e-7, explode_threshold=1e3, track_history=True):
        self.mp = mp
        self.params = params
        self.vanish_threshold = vanish_threshold
        self.explode_threshold = explode_threshold
        self.track_history = track_history

        # Statistics tracking
        self.step_count = 0
        self.history = [] if track_history else None

        # Counters for issues
        self.vanishing_count = 0
        self.exploding_count = 0
        self.vanishing_params = set()
        self.exploding_params = set()

    def compute_grad_norm(self, grad):
        """Compute L2 norm of a gradient tensor."""
        return float(self.mp.sqrt(self.mp.sum(grad * grad)))

    def compute_global_norm(self, grads):
        """Compute global L2 norm across all gradients."""
        total_norm_sq = 0.0
        for g in grads:
            total_norm_sq += float(self.mp.sum(g * g))
        return total_norm_sq ** 0.5

    def analyze(self, grads, param_names=None):
        """
        Analyze gradients for vanishing/exploding issues.

        Args:
            grads: List of gradient tensors
            param_names: Optional list of parameter names for reporting

        Returns:
            dict with analysis results
        """
        self.step_count += 1

        if param_names is None:
            param_names = [f"param_{i}" for i in range(len(grads))]

        # Per-parameter analysis
        param_stats = []
        vanishing_this_step = []
        exploding_this_step = []

        for i, (g, name) in enumerate(zip(grads, param_names)):
            norm = self.compute_grad_norm(g)
            mean = float(self.mp.mean(g))
            std = float(self.mp.std(g)) if hasattr(self.mp, 'std') else float(self.mp.sqrt(self.mp.mean((g - mean) ** 2)))
            max_val = float(self.mp.max(self.mp.abs(g)))
            min_val = float(self.mp.min(self.mp.abs(g)))

            status = "normal"
            if norm < self.vanish_threshold:
                status = "vanishing"
                vanishing_this_step.append(name)
                self.vanishing_params.add(name)
                self.vanishing_count += 1
            elif norm > self.explode_threshold or self.mp.any(self.mp.isnan(g)) or self.mp.any(self.mp.isinf(g)):
                status = "exploding"
                exploding_this_step.append(name)
                self.exploding_params.add(name)
                self.exploding_count += 1

            param_stats.append({
                "name": name,
                "norm": norm,
                "mean": mean,
                "std": std,
                "max": max_val,
                "min": min_val,
                "status": status,
                "shape": g.shape
            })

        # Global statistics
        global_norm = self.compute_global_norm(grads)
        norms = [s["norm"] for s in param_stats]

        result = {
            "step": self.step_count,
            "global_norm": global_norm,
            "max_norm": max(norms) if norms else 0,
            "min_norm": min(norms) if norms else 0,
            "mean_norm": sum(norms) / len(norms) if norms else 0,
            "vanishing_params": vanishing_this_step,
            "exploding_params": exploding_this_step,
            "has_vanishing": len(vanishing_this_step) > 0,
            "has_exploding": len(exploding_this_step) > 0,
            "param_stats": param_stats
        }

        if self.track_history:
            # Store summary only to avoid memory issues
            self.history.append({
                "step": self.step_count,
                "global_norm": global_norm,
                "max_norm": max(norms) if norms else 0,
                "min_norm": min(norms) if norms else 0,
                "has_vanishing": result["has_vanishing"],
                "has_exploding": result["has_exploding"]
            })

        return result

    def check(self, grads):
        """
        Quick check for gradient issues without full analysis.

        Returns:
            tuple: (has_issue, issue_type) where issue_type is 'vanishing', 'exploding', or None
        """
        global_norm = self.compute_global_norm(grads)

        # Check for NaN or Inf
        for g in grads:
            if self.mp.any(self.mp.isnan(g)) or self.mp.any(self.mp.isinf(g)):
                return True, "exploding"

        if global_norm < self.vanish_threshold:
            return True, "vanishing"
        elif global_norm > self.explode_threshold:
            return True, "exploding"

        return False, None

    def get_summary(self):
        """Get a summary of gradient health across all monitored steps."""
        return {
            "total_steps": self.step_count,
            "vanishing_incidents": self.vanishing_count,
            "exploding_incidents": self.exploding_count,
            "problematic_params_vanishing": list(self.vanishing_params),
            "problematic_params_exploding": list(self.exploding_params),
            "health_score": 1.0 - (self.vanishing_count + self.exploding_count) / max(self.step_count * len(self.params), 1)
        }

    def get_layer_gradient_flow(self, grads, param_names=None):
        """
        Analyze gradient flow across layers to detect where vanishing/exploding starts.
        Useful for diagnosing which layers are problematic.

        Returns:
            List of (layer_name, norm) tuples ordered by gradient flow
        """
        if param_names is None:
            param_names = [f"param_{i}" for i in range(len(grads))]

        flow = []
        for g, name in zip(grads, param_names):
            norm = self.compute_grad_norm(g)
            flow.append((name, norm))

        return flow

    def reset(self):
        """Reset all statistics."""
        self.step_count = 0
        self.history = [] if self.track_history else None
        self.vanishing_count = 0
        self.exploding_count = 0
        self.vanishing_params = set()
        self.exploding_params = set()
