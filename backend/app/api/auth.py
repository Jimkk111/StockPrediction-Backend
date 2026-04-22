from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.user import User
from app.utils.security import get_current_user, verify_password, create_access_token
from app.schemas.user import UserRegister, Token, UserResponse
from app.services.auth import register_user, get_user_profile

router = APIRouter(prefix="/auth", tags=["用户鉴权"])


@router.post("/register", response_model=UserResponse, summary="用户注册")
def register(user_data: UserRegister, db: Session = Depends(get_db)):
    return register_user(db, user_data)


@router.post("/login", response_model=Token, summary="用户登录")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户名或密码错误")
    if not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户名或密码错误")
    access_token = create_access_token(data={"sub": user.username})
    return Token(access_token=access_token, token_type="bearer")


@router.get("/me", response_model=UserResponse, summary="获取当前用户信息")
def read_current_user(current_user: User = Depends(get_current_user)):
    return UserResponse.model_validate(current_user)
