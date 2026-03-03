"""
Configuration management for Intuition-Driven Market Prediction AI.
Centralizes all configurable parameters with type safety and validation.
"""

import os
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class DataConfig:
    """Data collection and preprocessing configuration"""
    # Data sources
    data_sources: List[str] = field(default_factory=lambda: [
        'yfinance',  # Yahoo Finance for traditional markets
        'ccxt',      # Cryptocurrency exchanges
        'alpaca',    # Commission-free trading API
    ])
    
    # Timeframes for multi-scale analysis
    timeframes: List[str] = field(default_factory=lambda: [
        '1h', '4h', '1d', '1w'
    ])
    
    # Historical data depth (adjust based on market cycles)
    lookback_periods: Dict[str, int] = field(default_factory=lambda: {
        '1h': 30,    # 30 days of hourly data
        '4h': 90,    # 90 days of 4-hour data  
        '1d': 365,   # 1 year of daily data
        '1w': 1040,  # 20 years of weekly data
    })
    
    # Feature engineering parameters
    technical_indicators: List[str] = field(default_factory=lambda: [
        'RSI', 'MACD', 'BBANDS', 'ATR', 
        'OBV', 'STOCH', 'ADX', 'CCI'
    ])
    
    # Market regime detection
    regime_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'volatility_regime': 0.25,  # 25% threshold for high volatility
        'trend_strength': 0.3,      # ADX threshold for trend detection
        'oversold_rsi': 30.0,
        'overbought_rsi': 70.0,
    })

@dataclass
class ModelConfig:
    """Deep learning model configuration for intuition learning"""
    # Model architecture
    lstm_units: List[int] = field(default_factory=lambda: [128, 64, 32])
    dropout_rate: float = 0.3
    attention_heads: int = 4
    
    # Training parameters
    sequence_length: int = 60  # Number of time steps in each sequence
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    early_stopping_patience: int = 15
    
    # Validation and testing
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    
    # Feature importance threshold
    feature_importance_threshold: float = 0.05

@dataclass
class TradingConfig:
    """Trading strategy and risk management configuration"""
    # Position sizing
    max_position_size: float = 0.1  # Max 10% of portfolio per trade
    max_portfolio_risk: float = 0.02  # Max 2% total portfolio risk
    
    # Stop loss and take profit
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.04  # 4% take profit
    trailing_stop_pct: float = 0.01  # 1% trailing stop
    
    # Trading constraints
    max_daily_trades: int = 5
    min_confidence_score: float = 0.65  # Minimum confidence to execute
    
    # Slippage and commission (realistic estimates)
    slippage_pct: float = 0.0005  # 0.05% slippage
    commission_pct: float = 0.001  # 0.1% commission

@dataclass
class FirebaseConfig:
    """Firebase configuration for real-time data streaming"""
    project_id: str = field(default_factory=lambda