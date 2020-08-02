from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC


def estimator(C=16, verbose=False, kernel='rbf', gamma=0.25, tol=0.0001, output_probability=False, class_weight='balanced'):
    classifier = SVC(
        C=C,
        kernel=kernel,
        gamma=gamma,
        tol=tol,
        probability=output_probability,
        class_weight=class_weight,
        verbose=verbose
    )
    scaler = MinMaxScaler((-1, 1))
    pipe = make_pipeline(scaler, classifier)
    return pipe
