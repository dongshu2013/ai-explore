from fastapi import Request, HTTPException
import time
from starlette.middleware.base import BaseHTTPMiddleware

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, requests_per_minute=60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests = {}

    async def dispatch(self, request: Request, call_next):
        # Get client IP safely, with fallback
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()
        
        # Clean old requests
        self.requests = {ip: times for ip, times in self.requests.items() 
                        if current_time - times[-1] < 60}
        
        if client_ip in self.requests:
            if len(self.requests[client_ip]) >= self.requests_per_minute:
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            self.requests[client_ip].append(current_time)
        else:
            self.requests[client_ip] = [current_time]
        
        response = await call_next(request)
        return response 