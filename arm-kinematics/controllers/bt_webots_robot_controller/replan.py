import py_trees

class ReplanIfRequested2(py_trees.behaviour.Behaviour):
  def __init__(self):
      super().__init__("ReplanIfRequested")
      self.blackboard = py_trees.blackboard.Client()
      self.blackboard.register_key("replan", access=py_trees.common.Access.READ_WRITE)
  
  def update(self):
      if self.blackboard.replan:
          print("[Replan] Triggered.")
          planner_fn()
          self.blackboard.replan_needed = False
          return py_trees.common.Status.SUCCESS
      else:
          return py_trees.common.Status.FAILURE