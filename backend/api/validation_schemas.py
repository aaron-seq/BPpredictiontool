"""
Validation schemas for API requests and responses.
"""
from marshmallow import Schema, fields, validate, ValidationError, post_load
from typing import List, Dict, Any
import numpy as np

class PredictionRequestSchema(Schema):
    """Schema for blood pressure prediction requests."""
    
    ppg_signal = fields.List(
        fields.Float(required=True),
        required=True,
        validate=validate.Length(equal=1000),
        metadata={'description': 'PPG signal data (1000 values)'}
    )
    
    ecg_signal = fields.List(
        fields.Float(required=True),
        required=True,
        validate=validate.Length(equal=1000),
        metadata={'description': 'ECG signal data (1000 values)'}
    )
    
    include_quality_metrics = fields.Boolean(
        missing=False,
        metadata={'description': 'Include signal quality assessment in response'}
    )
    
    @post_load
    def validate_signals(self, data, **kwargs):
        """Additional validation for signal data."""
        ppg = np.array(data['ppg_signal'])
        ecg = np.array(data['ecg_signal'])
        
        # Check for NaN or infinite values
        if np.any(np.isnan(ppg)) or np.any(np.isinf(ppg)):
            raise ValidationError('PPG signal contains invalid values (NaN or Inf)')
        
        if np.any(np.isnan(ecg)) or np.any(np.isinf(ecg)):
            raise ValidationError('ECG signal contains invalid values (NaN or Inf)')
        
        # Check signal range (reasonable physiological values)
        if np.max(np.abs(ppg)) > 1000:
            raise ValidationError('PPG signal values are out of reasonable range')
        
        if np.max(np.abs(ecg)) > 1000:
            raise ValidationError('ECG signal values are out of reasonable range')
        
        return data

class PredictionResponseSchema(Schema):
    """Schema for blood pressure prediction responses."""
    
    predicted_abp = fields.List(
        fields.Float(),
        required=True,
        metadata={'description': 'Predicted arterial blood pressure waveform'}
    )
    
    systolic_bp = fields.Float(
        required=True,
        metadata={'description': 'Predicted systolic blood pressure (mmHg)'}
    )
    
    diastolic_bp = fields.Float(
        required=True,
        metadata={'description': 'Predicted diastolic blood pressure (mmHg)'}
    )
    
    mean_arterial_pressure = fields.Float(
        required=True,
        metadata={'description': 'Predicted mean arterial pressure (mmHg)'}
    )
    
    confidence_score = fields.Float(
        required=True,
        validate=validate.Range(min=0, max=1),
        metadata={'description': 'Prediction confidence score (0-1)'}
    )
    
    quality_metrics = fields.Dict(
        missing=None,
        metadata={'description': 'Signal quality assessment metrics'}
    )

class ErrorResponseSchema(Schema):
    """Schema for error responses."""
    
    error = fields.String(
        required=True,
        metadata={'description': 'Error message'}
    )
    
    error_code = fields.String(
        required=True,
        metadata={'description': 'Error code'}
    )
    
    details = fields.Dict(
        missing=None,
        metadata={'description': 'Additional error details'}
    )

class HealthCheckResponseSchema(Schema):
    """Schema for health check responses."""
    
    status = fields.String(
        required=True,
        validate=validate.OneOf(['healthy', 'unhealthy']),
        metadata={'description': 'Service health status'}
    )
    
    model_loaded = fields.Boolean(
        required=True,
        metadata={'description': 'Whether the ML model is loaded'}
    )
    
    timestamp = fields.DateTime(
        required=True,
        metadata={'description': 'Health check timestamp'}
    )
    
    version = fields.String(
        required=True,
        metadata={'description': 'API version'}
    )

# Schema instances
prediction_request_schema = PredictionRequestSchema()
prediction_response_schema = PredictionResponseSchema()
error_response_schema = ErrorResponseSchema()
health_check_response_schema = HealthCheckResponseSchema()
