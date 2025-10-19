from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from routers.profile import Profile

class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        user_id = request.cookies.get("userId")
        if user_id:
            user = await Profile.get(user_id)
            if user:
                request.state.user = user.dict()
        response = await call_next(request)
        return response
