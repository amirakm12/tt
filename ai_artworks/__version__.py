"""
AI-ARTWORKS Enterprise Edition
Version and build information for production deployment
"""

__version__ = "1.0.0"
__version_info__ = (1, 0, 0)
__release__ = "stable"
__build__ = "20250103-enterprise"
__edition__ = "Enterprise"

# Enterprise features
__features__ = {
    "multi_tenant": True,
    "sso_enabled": True,
    "audit_logging": True,
    "high_availability": True,
    "clustering": True,
    "enterprise_support": True,
    "sla_guarantee": "99.99%",
    "max_users": "unlimited",
    "api_rate_limit": "10000/min",
    "data_encryption": "AES-256",
    "compliance": ["SOC2", "HIPAA", "GDPR", "ISO27001"]
}

# License information
__license__ = "Commercial Enterprise License"
__license_key__ = None  # Set during installation
__support_tier__ = "24/7 Premium"