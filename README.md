# water-systems-gym
A repository containing two OpenAI Gym environments for Reinforcement learning. The environments represent basic water systems with objects like water tanks and pumps, etc., where agents can learn to predict and control processes involved in water distribution networks. 


## Development setup

	virtualenv -p python3.7 env
	source env/bin/activate
	pip install -r requirements.txt
	python setup.py develop

Create `.env` file in root path

For using PPO with stable-baselines:
	source env/bin/activate
	pip install stable-baselines[mpi]

### Run tests

	./manage.py test test_files

### MLflow

 For usage of MLflow server add to `.env`:
	MLFLOW_USER=<MLFLOW_USER>
	MLFLOW_PASSWORD=<MLFLOW_PASSWORD>

 ### RL examples
 	For optimization with scipy algorithms run generate_optimize_scenes from analytics/reinforecement_learinig module:
	
	cd analytics/pump_gym

	python reinforcement_learning_module --params <hyperparameter file>
	                                     -- nscenes <number of scenes to generate>
	                                     --seed <Random seed> 
                                         -- nproc<Number of processes to run>
	                                     --scenes and --results <Names of scene and result files>
	Example pump_gym:
		python generate_optimize_scenes.py --params anytownMasterRandom --nscenes 10 --seed 7 --scenes anytown_scenes_random --result anytown_result_random
		python train.py 


	cd analytics/tank_plc_gym
	
	Example tank_plc_gym:
		python train.py
		python rllib_train.py

### dvc commands
	Set dvc with azure connection string
		dvc remote modify --local myremote connection_string "$AZURE_CONNECTION_STRING"

	dvc add <filename> creates <filename>.dvc file
	git add <filename>.dvc to add file to git
	git rm -r --cached <filename> if file was already tracked by git
	dvc remove <filename>.dvc stop tracking file
	dvc add -R <foldername> add every file from folder

### Shell

Running plan 'ipython' with activated environment will work, too. This however
imports SQLAlchemy session, sanic app and the models:

	source env/bin/activate
	./manage.py shell
