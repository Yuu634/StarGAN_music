import math
from dataclasses import dataclass, field
from typing import Dict, Optional

@dataclass
class MonitoringMetrics:
    """Monitoring metrics for training stability"""
    d_loss: float = 0.0
    g_loss: float = 0.0
    d_loss_ema: float = 0.0
    g_loss_ema: float = 0.0
    d_grad_norm: float = 0.0
    g_grad_norm: float = 0.0
    balance_ratio: float = 1.0  # D_loss / G_loss
    d_ideal_gap: float = 0.0    # |D_loss - ideal_d_loss|
    stability_score: float = 1.0  # 0.0 ~ 1.0, higher is better
    lr_multiplier: float = 1.0  # Learning rate adjustment factor
    
    def __post_init__(self):
        self.ideal_d_loss = math.log(4)  # For non-saturating GAN


class AdaptiveHyperparameterScheduler:
    """
    Adaptive hyperparameter scheduler based on Gap-Aware Learning Rate Scheduler
    and training stability metrics.
    """
    
    def __init__(
        self,
        initial_g_lr: float = 1e-4,
        initial_d_lr: float = 1e-4,
        initial_lambda_gp: float = 10.0,
        initial_lambda_cls: float = 1.0,
        initial_lambda_rec: float = 10.0,
        ideal_d_loss: float = 0.0,  # Default: log(4) for non-saturating GAN
        ideal_balance_ratio: float = 1.5,  # Target D_loss / G_loss ratio
        ema_decay: float = 0.95,
        stability_threshold: float = 0.5,
        warmup_steps: int = 5000,
    ):
        """
        Args:
            initial_g_lr: Initial generator learning rate
            initial_d_lr: Initial discriminator learning rate
            initial_lambda_gp: Initial gradient penalty weight
            initial_lambda_cls: Initial classification loss weight
            initial_lambda_rec: Initial reconstruction loss weight
            ideal_d_loss: Ideal discriminator loss (default: log(4) for non-saturating GAN)
            ideal_balance_ratio: Target balance ratio (D_loss / G_loss)
            ema_decay: Exponential moving average decay factor
            stability_threshold: Threshold for stability score below which warnings are issued
            warmup_steps: Number of warmup steps before enabling adaptive adjustment
        """
        self.initial_g_lr = initial_g_lr
        self.initial_d_lr = initial_d_lr
        self.initial_lambda_gp = initial_lambda_gp
        self.initial_lambda_cls = initial_lambda_cls
        self.initial_lambda_rec = initial_lambda_rec
        
        self.current_g_lr = initial_g_lr
        self.current_d_lr = initial_d_lr
        self.current_lambda_gp = initial_lambda_gp
        self.current_lambda_cls = initial_lambda_cls
        self.current_lambda_rec = initial_lambda_rec
        
        self.ideal_d_loss = ideal_d_loss
        self.ideal_balance_ratio = ideal_balance_ratio
        self.ema_decay = ema_decay
        self.stability_threshold = stability_threshold
        self.warmup_steps = warmup_steps
        
        # Exponential moving averages
        self.d_loss_ema = None
        self.g_loss_ema = None
        
        # History for trend analysis
        self.loss_history = {'d_loss': [], 'g_loss': [], 'balance_ratio': []}
        self.lr_history = {'g_lr': [], 'd_lr': []}
        self.stability_history = []
        
        # Step counter
        self.step_count = 0
        
        # Bounds for hyperparameter adjustments
        self.g_lr_bounds = (initial_g_lr * 0.1, initial_g_lr * 3.0)
        self.d_lr_bounds = (initial_d_lr * 0.1, initial_d_lr * 3.0)
        self.lambda_gp_bounds = (initial_lambda_gp * 0.1, initial_lambda_gp * 2.0)
        self.lambda_cls_bounds = (initial_lambda_cls * 0.5, initial_lambda_cls * 2.0)
        self.lambda_rec_bounds = (initial_lambda_rec * 0.5, initial_lambda_rec * 2.0)
    
    def update(
        self,
        d_loss: float,
        g_loss: float,
        d_grad_norm: Optional[float] = None,
        g_grad_norm: Optional[float] = None,
    ) -> MonitoringMetrics:
        """
        Update hyperparameters based on current losses and gradient norms.
        
        Args:
            d_loss: Discriminator loss
            g_loss: Generator loss
            d_grad_norm: Discriminator gradient norm
            g_grad_norm: Generator gradient norm
        
        Returns:
            MonitoringMetrics object with current state
        """
        # Update EMA of losses
        if self.d_loss_ema is None:
            self.d_loss_ema = d_loss
            self.g_loss_ema = g_loss
        else:
            self.d_loss_ema = self.ema_decay * self.d_loss_ema + (1 - self.ema_decay) * d_loss
            self.g_loss_ema = self.ema_decay * self.g_loss_ema + (1 - self.ema_decay) * g_loss
        
        # Calculate metrics
        balance_ratio = self.d_loss_ema / max(self.g_loss_ema, 1e-7)
        d_ideal_gap = abs(self.d_loss_ema - self.ideal_d_loss)
        
        # Calculate stability score (0.0 ~ 1.0)
        stability_score = self._calculate_stability_score(
            balance_ratio, d_ideal_gap, d_grad_norm, g_grad_norm
        )
        
        # Calculate learning rate multiplier (only after warmup)
        lr_multiplier = self._calculate_lr_multiplier(
            d_ideal_gap, balance_ratio
        ) if self.step_count > self.warmup_steps else 1.0
        
        # Update hyperparameters
        if self.step_count > self.warmup_steps:
            self._update_learning_rates(lr_multiplier, balance_ratio)
            self._update_lambda_values(stability_score, balance_ratio)
        
        # Create metrics object
        metrics = MonitoringMetrics(
            d_loss=d_loss,
            g_loss=g_loss,
            d_loss_ema=self.d_loss_ema,
            g_loss_ema=self.g_loss_ema,
            d_grad_norm=d_grad_norm or 0.0,
            g_grad_norm=g_grad_norm or 0.0,
            balance_ratio=balance_ratio,
            d_ideal_gap=d_ideal_gap,
            stability_score=stability_score,
            lr_multiplier=lr_multiplier,
        )
        
        # Update history
        self.loss_history['d_loss'].append(d_loss)
        self.loss_history['g_loss'].append(g_loss)
        self.loss_history['balance_ratio'].append(balance_ratio)
        self.lr_history['g_lr'].append(self.current_g_lr)
        self.lr_history['d_lr'].append(self.current_d_lr)
        self.stability_history.append(stability_score)
        
        self.step_count += 1
        
        return metrics
    
    def _calculate_stability_score(
        self,
        balance_ratio: float,
        d_ideal_gap: float,
        d_grad_norm: Optional[float],
        g_grad_norm: Optional[float],
    ) -> float:
        """
        Calculate overall training stability score (0.0 ~ 1.0)
        Higher score means more stable training.
        """
        # Balance score: penalize if ratio deviates from ideal
        balance_error = abs(balance_ratio - self.ideal_balance_ratio) / max(self.ideal_balance_ratio, 1e-7)
        balance_score = max(0.0, 1.0 - balance_error)
        
        # Gap score: penalize if D_loss deviates from ideal
        gap_error = d_ideal_gap / max(self.ideal_d_loss, 1e-7)
        gap_score = max(0.0, 1.0 - gap_error * 0.5)  # Less weight than balance
        
        # Gradient score: penalize if gradients are too large or NaN
        grad_score = 1.0
        if d_grad_norm is not None and g_grad_norm is not None:
            max_grad = max(d_grad_norm, g_grad_norm)
            if max_grad > 10.0:  # Gradient explosion threshold
                grad_score = max(0.1, 1.0 - (max_grad - 10.0) / 100.0)
            elif max_grad > 1.0:
                grad_score = 0.9
        
        # Combine scores
        stability_score = 0.5 * balance_score + 0.3 * gap_score + 0.2 * grad_score
        return float(max(0.0, min(1.0, stability_score)))
    
    def _calculate_lr_multiplier(
        self,
        d_ideal_gap: float,
        balance_ratio: float,
    ) -> float:
        """
        Calculate learning rate adjustment multiplier based on Gap-Aware scheduler.
        
        Gap-Aware idea:
        - If D_loss > ideal_d_loss: increase D_lr (discriminator too weak)
        - If D_loss < ideal_d_loss: decrease D_lr (discriminator too strong)
        - Balance control: if ratio >> ideal, boost generator; if ratio << ideal, boost discriminator
        """
        # Gap-based adjustment for D_lr
        gap_normalized = d_ideal_gap / max(self.ideal_d_loss, 1e-7)
        if self.d_loss_ema > self.ideal_d_loss:
            # D_loss too high: boost D_lr
            d_gap_factor = 1.0 + 0.5 * min(gap_normalized, 1.0)
        else:
            # D_loss too low: reduce D_lr
            d_gap_factor = max(0.5, 1.0 - 0.5 * gap_normalized)
        
        # Balance-based adjustment for G_lr
        balance_error = (balance_ratio - self.ideal_balance_ratio) / max(self.ideal_balance_ratio, 1e-7)
        if balance_ratio > self.ideal_balance_ratio:
            # D dominates: reduce D_lr or increase G_lr
            g_balance_factor = 1.0 + 0.3 * min(abs(balance_error), 1.0)
        else:
            # G dominates: increase D_lr
            g_balance_factor = max(0.8, 1.0 - 0.2 * min(abs(balance_error), 1.0))
        
        # Average multiplier (will be applied separately)
        multiplier = 0.5 * d_gap_factor + 0.5 * g_balance_factor
        return float(max(0.5, min(2.0, multiplier)))
    
    def _update_learning_rates(
        self,
        lr_multiplier: float,
        balance_ratio: float,
    ):
        """Update learning rates based on multiplier and balance ratio"""
        # Gap-aware adjustment
        if self.d_loss_ema is not None:
            gap_normalized = abs(self.d_loss_ema - self.ideal_d_loss) / max(self.ideal_d_loss, 1e-7)
            
            # D_lr adjustment based on gap
            if self.d_loss_ema > self.ideal_d_loss:
                d_lr_new = self.current_d_lr * (1.0 + 0.1 * min(gap_normalized, 1.0))
            else:
                d_lr_new = self.current_d_lr * max(0.9, 1.0 - 0.1 * gap_normalized)
            
            # G_lr adjustment based on balance
            if balance_ratio > self.ideal_balance_ratio:
                g_lr_new = self.current_g_lr * 1.05  # Boost G slightly
            else:
                g_lr_new = self.current_g_lr * 0.98  # Reduce G slightly
        else:
            d_lr_new = self.current_d_lr * lr_multiplier
            g_lr_new = self.current_g_lr * lr_multiplier
        
        # Apply bounds
        self.current_d_lr = float(max(self.d_lr_bounds[0], min(self.d_lr_bounds[1], d_lr_new)))
        self.current_g_lr = float(max(self.g_lr_bounds[0], min(self.g_lr_bounds[1], g_lr_new)))
    
    def _update_lambda_values(
        self,
        stability_score: float,
        balance_ratio: float,
    ):
        """Update loss weights based on stability and balance"""
        # Conservative adjustment: only update if stability is compromised
        if stability_score < 0.7:
            # Training is unstable: reduce lambda values to stabilize
            adjustment_factor = 0.95 + 0.05 * stability_score  # 0.95 ~ 1.0
            self.current_lambda_gp *= adjustment_factor
            self.current_lambda_cls *= adjustment_factor
            self.current_lambda_rec *= adjustment_factor
        elif stability_score > 0.9:
            # Training is very stable: can afford to increase weights
            adjustment_factor = 1.02
            self.current_lambda_gp *= adjustment_factor
            self.current_lambda_cls *= adjustment_factor
            self.current_lambda_rec *= adjustment_factor
        
        # Balance-based lambda adjustments
        if balance_ratio < 0.8:
            # G too strong: increase D weights (lambda_cls affects D strength)
            self.current_lambda_cls = min(self.lambda_cls_bounds[1], self.current_lambda_cls * 1.02)
        elif balance_ratio > 2.5:
            # D too strong: reduce D weights
            self.current_lambda_cls = max(self.lambda_cls_bounds[0], self.current_lambda_cls * 0.98)
        
        # Apply bounds
        self.current_lambda_gp = float(max(self.lambda_gp_bounds[0], min(self.lambda_gp_bounds[1], self.current_lambda_gp)))
        self.current_lambda_cls = float(max(self.lambda_cls_bounds[0], min(self.lambda_cls_bounds[1], self.current_lambda_cls)))
        self.current_lambda_rec = float(max(self.lambda_rec_bounds[0], min(self.lambda_rec_bounds[1], self.current_lambda_rec)))
    
    def get_state_dict(self) -> Dict:
        """Get scheduler state for checkpointing"""
        return {
            'step_count': self.step_count,
            'd_loss_ema': self.d_loss_ema,
            'g_loss_ema': self.g_loss_ema,
            'current_g_lr': self.current_g_lr,
            'current_d_lr': self.current_d_lr,
            'current_lambda_gp': self.current_lambda_gp,
            'current_lambda_cls': self.current_lambda_cls,
            'current_lambda_rec': self.current_lambda_rec,
            'loss_history': self.loss_history,
            'lr_history': self.lr_history,
            'stability_history': self.stability_history,
        }
    
    def load_state_dict(self, state_dict: Dict):
        """Load scheduler state from checkpoint"""
        self.step_count = state_dict['step_count']
        self.d_loss_ema = state_dict['d_loss_ema']
        self.g_loss_ema = state_dict['g_loss_ema']
        self.current_g_lr = state_dict['current_g_lr']
        self.current_d_lr = state_dict['current_d_lr']
        self.current_lambda_gp = state_dict['current_lambda_gp']
        self.current_lambda_cls = state_dict['current_lambda_cls']
        self.current_lambda_rec = state_dict['current_lambda_rec']
        self.loss_history = state_dict.get('loss_history', self.loss_history)
        self.lr_history = state_dict.get('lr_history', self.lr_history)
        self.stability_history = state_dict.get('stability_history', self.stability_history)
