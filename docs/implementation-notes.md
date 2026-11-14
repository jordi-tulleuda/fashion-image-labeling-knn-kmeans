# Implementation and provenance notes

## Group coursework (UAB, 2025–26)

The submitted project contains two core models (`KNN.py`, `Kmeans.py`), integration and evaluation (`my_labeling.py`), and utilities (`utils.py`, `utils_data.py`). The original work was completed in a team of three. The README distinguishes documented individual and shared contributions.

### Jordi's documented contribution

The project report's **“Millores Jordi”** section and the student presentation explicitly attribute to Jordi the exploration of three distances in the KNN classifier: Euclidean, Manhattan (`cityblock`) and cosine. The code also labels `Retrieval_by_shape` and `get_shape_accuracy` as Jordi's analysis functions. The reported study evaluated odd K values from 1 through 25. See the README chart for the original results (the chart is reproduced from the submitted team report, not rerun here).

Other improvements in the team report (centroid initialization, K-means convergence controls and the combined retrieval pipeline) are team contributions or documented as other members' work. They are not presented as Jordi's personal implementations.

## Portfolio maintenance changes

Compared with the supplied ZIP:
- Removed the Windows virtual environment, IDE settings, duplicated `*_antic.py` files and source code containing student ID numbers.
- Omitted university pickle fixtures, quizzes and the third-party image dataset; those are not necessary for the self-contained synthetic demo.
- Added `demo.py` and independent tests using synthetic data.
- Restored the missing `distance` function referenced by the original test suite.
- Fixed `KMeans.find_bestK`: the submitted version returned from inside the candidate loop and could not assess the requested range. The corrected routine compares all candidates, selects via a basic WCD elbow rule or the Fisher/Inter separation metric, then fits the chosen K.

**Caution:** These portfolio fixes change the original K-means implementation and are not asserted to reproduce the coursework's numerical experiments. The full historical analysis requires the UAB course dataset and a separate reproducibility run.

## Original quiz scripts

`prova_quizz_kmeans.py` and `quizzCases-kmeans.py` were found in the supplied ZIP. They are small ad-hoc scripts, with the latter running six scenarios using images supplied for a course quiz. They were omitted from the public-facing repository to keep the main code clean and avoid distributing course quiz images; the self-contained `tests/` suite is a stronger portfolio demonstration.

## Redistribution

The repository includes utility modules originally provided for UAB coursework, as well as collaboratively developed code. Confirm both the department's redistribution policy and your teammates' agreement before changing a new repository from private to public. No license is asserted for the shared materials.
