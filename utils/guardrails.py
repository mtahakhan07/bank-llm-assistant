import re
from typing import List, Dict, Any, Optional

# Define the sensitive information patterns
SENSITIVE_PATTERNS = [
    r"password",
    r"credentials",
    r"account\s+number",
    r"pin\s+code",
    r"social\s+security",
    r"credit\s+card",
    r"cvv",
    r"expiry\s+date",
    r"routing\s+number",
    r"authorization\s+code",
    r"private\s+key",
    r"secret\s+key",
    r"confidential",
    r"classified",
]

# Define harmful or inappropriate content patterns
HARMFUL_PATTERNS = [
    r"hack",
    r"steal",
    r"fraud",
    r"illegal",
    r"launder",
    r"attack",
    r"exploit",
    r"bypass\s+security",
    r"circumvent",
    r"forge",
    r"counterfeit",
    r"unauthorized\s+access",
]

# Define prompt injection patterns
INJECTION_PATTERNS = [
    r"ignore\s+previous\s+instructions",
    r"disregard\s+(all|your)\s+training",
    r"forget\s+(all|your)\s+training",
    r"new\s+instruction",
    r"override",
]

def apply_guardrails(query: str) -> bool:
    """
    Apply guardrails to ensure the query is safe and appropriate.
    
    Args:
        query: The user's query
        
    Returns:
        True if the query passes all guardrails, False otherwise
    """
    # Convert to lowercase for case-insensitive matching
    query_lower = query.lower()
    
    # Check for sensitive information requests
    for pattern in SENSITIVE_PATTERNS:
        if re.search(pattern, query_lower):
            return False
    
    # Check for harmful or inappropriate content
    for pattern in HARMFUL_PATTERNS:
        if re.search(pattern, query_lower):
            return False
    
    # Check for prompt injection attempts
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, query_lower):
            return False
    
    # If passes all checks, return True
    return True

def filter_response(response: str) -> str:
    """
    Filter the response to ensure it doesn't contain sensitive information.
    
    Args:
        response: The generated response
        
    Returns:
        Filtered response with sensitive information removed
    """
    response_lower = response.lower()
    
    # Check for sensitive information leaks
    for pattern in SENSITIVE_PATTERNS:
        if re.search(pattern, response_lower):
            response = re.sub(r'(?i)\b(\w*' + pattern + r'\w*\s*:?\s*)[^\s.,;!?]*', r'\1[REDACTED]', response)
    
    return response

def detect_out_of_domain(query: str) -> bool:
    """
    Detect if the query is outside the domain of banking.
    
    Args:
        query: The user's query
        
    Returns:
        True if the query is out of domain, False otherwise
    """
    # List of keywords related to banking
    banking_keywords = [
        "account", "bank", "credit", "debit", "loan", "mortgage", "interest", 
        "deposit", "withdraw", "payment", "transfer", "balance", "transaction",
        "fee", "charge", "card", "online", "branch", "atm", "savings", "checking",
        "nust bank", "bank", "statement", "invest", "finance", "money", "cash",
        "rate", "term", "application", "approval", "open", "close", "service", "product"
    ]
    
    # Check if the query contains any banking keywords
    query_lower = query.lower()
    for keyword in banking_keywords:
        if keyword in query_lower:
            return False
    
    # If no banking keywords found, it might be out of domain
    return True 