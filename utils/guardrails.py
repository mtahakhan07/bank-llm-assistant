import re
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import sys

class GuardRails:
    """
    Guard rails for ensuring data privacy and security.
    """
    
    def __init__(self):
        """Initialize guard rails."""
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        
        # Regular expressions for sensitive data - only keep critical patterns
        self.patterns = {
            'ssn': r'\b\d{3}[-]?\d{2}[-]?\d{4}\b',
            'credit_card': r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
            'routing_number': r'\b\d{9}\b',
            'swift_code': r'\b[A-Z]{6}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b'
        }
        
        # Banking-specific keywords
        self.banking_keywords = {
            'accounts': ['checking', 'savings', 'investment', 'retirement', 'deposit', 'withdrawal'],
            'services': ['transfer', 'payment', 'loan', 'mortgage', 'credit', 'debit', 'wire'],
            'products': ['card', 'insurance', 'investment', 'deposit', 'account', 'loan'],
            'operations': ['withdraw', 'deposit', 'balance', 'statement', 'transaction', 'fee']
        }
        
        # Security patterns - only keep critical security checks
        self.security_patterns = {
            'sql_injection': r'(?i)(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER|CREATE)',
            'xss': r'(?i)(<script|javascript:|on\w+\s*=)',
            'command_injection': r'(?i)(;|\||&|>|<|`|\$|\(|\))'
        }
        
        # Inappropriate content patterns - only keep harmful patterns
        self.inappropriate_patterns = {
            'harmful': r'(?i)(hack|crack|exploit|bypass|unauthorized|illegal|fraud|scam|phishing|malware)',
            'sensitive': r'(?i)(password|secret|key|token|credential|auth)'
        }
    
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            stream=sys.stdout  # Force UTF-8 encoding
        )
    
    def filter_response(self, response: str) -> str:
        """
        Filter response for sensitive information and inappropriate content.
        Only blocks responses containing critical sensitive data or harmful content.
        
        Args:
            response: Response text to filter
            
        Returns:
            Filtered response
        """
        try:
            # Check for critical sensitive data
            for pattern_type, pattern in self.patterns.items():
                if re.search(pattern, response):
                    self.logger.warning(f"Critical sensitive {pattern_type} data found in response")
                    return "I apologize, but I cannot share that information as it contains sensitive financial data."
            
            # Check for harmful content
            if self._contains_inappropriate_content(response):
                self.logger.warning("Harmful content found in response")
                return "I apologize, but I cannot provide that information as it may be harmful."
            
            # Check for security issues
            if self._contains_security_issues(response):
                self.logger.warning("Security issues found in response")
                return "I apologize, but I cannot provide that information for security reasons."
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error filtering response: {str(e)}")
            return response  # Return original response on error instead of blocking it
    
    def detect_out_of_domain(self, query: str) -> bool:
        """
        Detect if a query is outside the banking domain.
        
        Args:
            query: Query text to check
            
        Returns:
            True if query is out of domain, False otherwise
        """
        try:
            # Convert query to lowercase for case-insensitive matching
            query_lower = query.lower()
            
            # Check if query contains any banking-related keywords
            for category, keywords in self.banking_keywords.items():
                if any(keyword in query_lower for keyword in keywords):
                    return False
            
            # Check for common banking-related phrases
            banking_phrases = [
                'bank', 'account', 'money', 'finance', 'payment',
                'transfer', 'loan', 'credit', 'debit', 'balance',
                'interest', 'rate', 'fee', 'transaction', 'statement',
                'savings', 'checking', 'deposit', 'withdrawal', 'atm',
                'online banking', 'mobile banking', 'branch', 'customer service',
                'nust bank', 'banking', 'financial', 'investment', 'mortgage',
                'insurance', 'wealth', 'retirement', 'pension', 'tax'
            ]
            
            # Check for banking-related question patterns
            banking_patterns = [
                r'how.*bank',
                r'what.*account',
                r'can.*open',
                r'need.*loan',
                r'want.*save',
                r'tell.*about.*bank',
                r'explain.*service',
                r'help.*with.*banking',
                r'information.*about.*bank',
                r'process.*of.*banking'
            ]
            
            # Check for banking phrases
            if any(phrase in query_lower for phrase in banking_phrases):
                return False
                
            # Check for banking patterns
            if any(re.search(pattern, query_lower) for pattern in banking_patterns):
                return False
            
            # If none of the above match, it's out of domain
            return True
            
        except Exception as e:
            self.logger.error(f"Error detecting out of domain: {str(e)}")
            return True  # Default to out of domain for safety
    
    def _contains_inappropriate_content(self, text: str) -> bool:
        """
        Check if text contains harmful content.
        
        Args:
            text: Text to check
            
        Returns:
            True if harmful content is found, False otherwise
        """
        # Convert text to lowercase for case-insensitive matching
        text_lower = text.lower()
        
        # Check for harmful patterns
        for category, patterns in self.inappropriate_patterns.items():
            if re.search(patterns, text_lower):
                self.logger.warning(f"Harmful {category} content found")
                return True
        
        return False
    
    def _contains_security_issues(self, text: str) -> bool:
        """
        Check if text contains security issues.
        
        Args:
            text: Text to check
            
        Returns:
            True if security issues are found, False otherwise
        """
        # Convert text to lowercase for case-insensitive matching
        text_lower = text.lower()
        
        # Check for security patterns
        for issue_type, pattern in self.security_patterns.items():
            if re.search(pattern, text_lower):
                self.logger.warning(f"Security issue found: {issue_type}")
                return True
        
        return False
    
    def validate_query(self, query: str) -> Dict[str, Any]:
        """
        Validate a query for security and appropriateness.
        
        Args:
            query: Query to validate
            
        Returns:
            Dictionary with validation results
        """
        try:
            # Check for sensitive data
            sensitive_data_found = False
            for pattern_type, pattern in self.patterns.items():
                if re.search(pattern, query):
                    sensitive_data_found = True
                    self.logger.warning(f"Sensitive {pattern_type} data found in query")
            
            # Check if out of domain
            is_out_of_domain = self.detect_out_of_domain(query)
            
            # Check for inappropriate content
            has_inappropriate_content = self._contains_inappropriate_content(query)
            
            # Check for security issues
            has_security_issues = self._contains_security_issues(query)
            
            return {
                'is_valid': not (sensitive_data_found or has_inappropriate_content or has_security_issues),
                'sensitive_data_found': sensitive_data_found,
                'is_out_of_domain': is_out_of_domain,
                'has_inappropriate_content': has_inappropriate_content,
                'has_security_issues': has_security_issues,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error validating query: {str(e)}")
            return {
                'is_valid': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def log_interaction(self, query: str, response: str, validation: Dict[str, Any]):
        """
        Log interaction for audit purposes.
        
        Args:
            query: User query
            response: System response
            validation: Validation results
        """
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'query': query,
                'response': response,
                'validation': validation
            }
            
            self.logger.info(f"Interaction logged: {log_entry}")
            
        except Exception as e:
            self.logger.error(f"Error logging interaction: {str(e)}")

# Create global instances for easy access
guardrails = GuardRails()

def filter_response(response: str) -> str:
    """Global function to filter response."""
    return guardrails.filter_response(response)

def detect_out_of_domain(query: str) -> bool:
    """Global function to detect out of domain queries."""
    return guardrails.detect_out_of_domain(query)

def validate_query(query: str) -> Dict[str, Any]:
    """Global function to validate query."""
    return guardrails.validate_query(query)

def log_interaction(query: str, response: str, validation: Dict[str, Any]):
    """Global function to log interaction."""
    guardrails.log_interaction(query, response, validation) 