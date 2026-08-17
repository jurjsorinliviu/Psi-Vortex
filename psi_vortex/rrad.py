"""Output and hidden-trajectory recurrent relation-aware distillation."""
from __future__ import annotations
import torch
from torch import nn
from .contracts import require_sequence


class RRADLoss(nn.Module):
    def __init__(self,teacher_hidden:int,student_hidden:int,alpha:float=1.,beta:float=.5,gamma:float=1.,delta:float=.5):
        super().__init__(); self.alpha,self.beta,self.gamma,self.delta=alpha,beta,gamma,delta
        self.teacher_projection=nn.Linear(teacher_hidden,student_hidden,bias=False)
    @staticmethod
    def _stat(a,b):
        difference=a-b; return (difference.square().sum(),difference.numel())
    def forward(self,student,teacher,time,student_hidden,teacher_hidden,previous:dict[str,torch.Tensor]|None=None):
        for name,value in (("student prediction",student),("teacher prediction",teacher),("student hidden",student_hidden),("teacher hidden",teacher_hidden)): require_sequence(value,name=name)
        projected=self.teacher_projection(teacher_hidden)
        relation_student,relation_teacher,relation_sh,relation_th,relation_time=student,teacher,student_hidden,projected,time
        if previous is not None:
            relation_student=torch.cat((previous["student"].detach(),student),1)
            relation_teacher=torch.cat((previous["teacher"].detach(),teacher),1)
            relation_sh=torch.cat((previous["student_hidden"].detach(),student_hidden),1)
            relation_th=torch.cat((self.teacher_projection(previous["teacher_hidden"].detach()),projected),1)
            relation_time=torch.cat((previous["time"].detach(),time),1)
        dt=relation_time[:,1:]-relation_time[:,:-1]
        if torch.any(dt<=0): raise ValueError("RRAD requires chronological time, including chunk boundaries")
        derivative=lambda value:(value[:,1:]-value[:,:-1])/dt
        stats={"output":self._stat(student,teacher),"output_relation":self._stat(derivative(relation_student),derivative(relation_teacher)),
               "hidden":self._stat(student_hidden,projected),"hidden_relation":self._stat(derivative(relation_sh),derivative(relation_th))}
        means={name:sse/count for name,(sse,count) in stats.items()}
        total=self.alpha*means["output"]+self.beta*means["output_relation"]+self.gamma*means["hidden"]+self.delta*means["hidden_relation"]
        means["statistics"]=stats
        return total,means

    def aggregate(self,statistics:list[dict[str,tuple[torch.Tensor,int]]]):
        means={name:sum(item[name][0] for item in statistics)/sum(item[name][1] for item in statistics) for name in ("output","output_relation","hidden","hidden_relation")}
        return self.alpha*means["output"]+self.beta*means["output_relation"]+self.gamma*means["hidden"]+self.delta*means["hidden_relation"]
