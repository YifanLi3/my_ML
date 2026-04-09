import pandera as pa
from pandera.typing import Series

from pydantic import BaseModel, Field

class InputSchema(pa.DataFrameModel):
    sepal_length: Series[float] = pa.Field(ge=0, le=20, coerce=True,title='sepal_length')
    sepal_width: Series[float] = pa.Field(ge=0, le=10, coerce=True,title='sepal_width')
    petal_length: Series[float] = pa.Field(ge=0, le=10, coerce=True,title='petal_length')
    petal_width: Series[float] = pa.Field(ge=0, le=5, coerce=True,title='petal_width')

    class Config:
        strict = True

#这是一个 pydantic, 通常用在 FastAPI 里定义 API 接口的请求体格式。
class IrisRequest(BaseModel):
    sepal_length: float = Field(..., ge=0, description='Sepal Length(cm)')
    sepal_width: float = Field(..., ge=0, description='Sepal Width(cm)')
    petal_length: float = Field(..., ge=0, description='Petal Length(cm)')
    petal_width: float = Field(..., ge=0, description='Petal Width(cm)')


class IrisResponse(BaseModel):
    prediction: str = Field(..., description='Predict Iris Species')
    probability: float = Field(..., description='Probability of prediction')
    confidence: str = Field(..., description='Confidence level(High/Medium/Low)' )