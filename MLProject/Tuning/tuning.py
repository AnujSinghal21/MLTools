### OptunaTuner
# !pip install optuna
import optuna
class OptunaTuner():
  def __init__(self, train, evaluate):
    self.params = {}
    self.train = train
    self.evaluate = evaluate
  def add_parameter(self, name, dtype='float', min=0.1, max=1.0, value=0):
    self.params[name] = {
        'dtype': dtype,
        'min': min,
        'max': max,
        'value': value
    }
  def get_objective(self):
    def objective(trial):
      train_hyperparams = {}
      for name, param in self.params.items():
        if param['dtype'] == 'int': train_hyperparams[name] = trial.suggest_int(name, param['min'], param['max'])
        elif param['dtype'] == 'float' : train_hyperparams[name] = trial.suggest_float(name, param['min'], param['max'])
        else: train_hyperparams[name] = param['value']
      model = self.train(**train_hyperparams)
      return self.evaluate(model)
    return objective
  def tune(self, iterations=100, direction='maximize'):
    self.study = optuna.create_study(direction=direction)
    objective = self.get_objective()
    self.study.optimize(objective, n_trials=iterations)
    print('**************** Tuning Complete ******************')
    params = self.study.best_trial.params
    value = self.study.best_trial.value
    print('BEST TRIAL VALUE:', value)
    print('BEST TRIAL PARAMETERS:', params)
  def get_tuned_model(self, retune=False):
    if (retune): self.tune()
    params = self.study.best_trial.params
    train_hyperparams = {name: param['value'] for name, param in self.params.items() }
    for name, val in params.items():
      train_hyperparams[name] = val
    model = self.train(**train_hyperparams)
    return model
