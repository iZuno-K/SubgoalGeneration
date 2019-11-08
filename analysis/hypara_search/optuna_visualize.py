import optuna
import matplotlib.pyplot as plt


def main():
    env = "Walker2d-v2"
    policy_mode = "large_variance"
    study = optuna.create_study(study_name='karino_{}_threshold_{}'.format(policy_mode, env),
                                storage='mysql://root@192.168.2.75/optuna', direction = "maximize", load_if_exists = True)

    df = study.trials_dataframe()
    df = df.sort_values(("intermediate_values", 1999))

    param = df['params'].values.squeeze()
    results = df[("intermediate_values", 1999)]

    plt.scatter(param, results)
    plt.show()


if __name__ == '__main__':
    main()
