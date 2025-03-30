from torch.nn.utils import (
  parameters_to_vector as Params2Vec,
  vector_to_parameters as Vec2Params
)

def tau_2d(alpha, beta, theta_ast):
  a = alpha * theta_ast[:,None,None]
  b = beta * alpha * theta_ast[:,None,None]
  return a + b

