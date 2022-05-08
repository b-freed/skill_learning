



def train():
    
    # loss_list_dict = 
    # for 
        # loss_dict = model.update(data,mode)
        # append losses
        # for k in loss_dict.keys():

    pass

def validate():

    pass

stage = # 'train_skills' or 'train_dynamics'

assert stage in ['train_skills','train_dynamics']

# make filenames for both
skill_filename = 
dynamics_filename = 

skill_path = 
dynamics_path = 

# initialize action-generative model
skill_model = SkillPolicyModel()  # TODO pass arguments
dymamics_model =   # TODO

model = TwoStageSkillModel(skill_model,dynamics_model,mode)


if stage == 'train_dynamics':
    # load the skill model from checkpoint
    skill_model.load_skills_from_ckpt(skill_path)





