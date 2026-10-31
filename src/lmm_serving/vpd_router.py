"""按空闲阶段槽位分配路由；调用方负责持有调度 Condition 的锁。"""


STAGES=("vision","prefill","decode")


class StageRouter:
    """一副本一次处理一个请求，V/P 交接后即可复用，D 保留到生成结束。"""

    def __init__(self,spec):
        self._pools={name:spec.ranks_for(stage) for stage,name in enumerate(STAGES)}
        self._cursors={name:0 for name in STAGES}
        self._owners={}
        self._routes={}
        self._released={}

    @property
    def owners(self):
        """返回快照，避免调度器外部修改槽位归属。"""
        return dict(self._owners)

    def acquire(self,request_id,has_images):
        """原子预留整条路由；没有足够槽位时返回 None，不推进轮询游标。"""
        if request_id in self._routes:
            raise ValueError("request already owns a route")
        route={"vision":None}
        choices={}
        for name in STAGES:
            if name=="vision" and not has_images:
                continue
            pool=self._pools[name]
            cursor=self._cursors[name]
            available=next(((cursor+offset)%len(pool) for offset in range(len(pool))
                            if pool[(cursor+offset)%len(pool)] not in self._owners),None)
            if available is None:
                return None
            route[name]=pool[available]
            choices[name]=available
        # 先确认三个阶段都可用，再统一提交，避免半条路由持有资源等待。
        # D 也提前预留：P 不会把 KV 发给仍在生成的 D；代价是活动路由数
        # 最多等于 D 副本数。更多已接纳请求留在主进程 FIFO，避免 GPU 排队持有 KV。
        for name,index in choices.items():
            self._owners[route[name]]=request_id
            self._cursors[name]=(index+1)%len(self._pools[name])
        self._routes[request_id]=route
        self._released[request_id]=set()
        return dict(route)

    def release_stage(self,request_id,stage_name):
        """在对应完成事件到达时释放一个槽位，不能重复释放或释放别人的槽位。"""
        route=self._routes[request_id]
        rank=route[stage_name]
        released=self._released[request_id]
        if rank is None or stage_name in released or self._owners.get(rank)!=request_id:
            raise RuntimeError("stage slot is not owned by this request")
        del self._owners[rank]
        released.add(stage_name)

    def finish(self,request_id):
        """正常结束请求；已释放的 V/P 可能正在服务别的请求，不能再次操作它们。"""
        route=self._routes[request_id]
        required={name for name,rank in route.items() if rank is not None}
        if self._released[request_id]!=required:
            raise RuntimeError("request still owns stage slots")
        del self._routes[request_id]
        del self._released[request_id]

    def abandon(self,request_id):
        """仅供交接前拒绝使用，回收本请求尚未释放的预留槽位。"""
        route=self._routes[request_id]
        remaining=[name for name,rank in route.items()
                   if rank is not None and name not in self._released[request_id]]
        for name in remaining:
            self.release_stage(request_id,name)
        self.finish(request_id)
