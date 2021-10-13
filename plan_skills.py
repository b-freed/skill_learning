'''File where we will sample a set of waypionts, and plan a sequence of skills to have our pointmass travel through those waypoits'''





for e in epochs:
  # Optimize plan: compute expected cost according to the current sequence of skills, take GD step on cost to optimize skills
  
  # Test plan: deploy learned skills in actual environment.  Now we're going be selecting base-level actions conditioned on the current skill and state, and executign that action in the real environment
