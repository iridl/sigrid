import numpy as np

from sigrid.harmonize import DisplayDataset
from sigrid.serve import pages


def test_build_tree_opens_path_to_variable():
    listings = [
        DisplayDataset(['NMME', 'SubC'], []),
        DisplayDataset(['ECCC', 'EMC'], []),
        DisplayDataset([], ['ua', 'tas']),
    ]
    root = pages.build_tree('/data', ['SubC', 'ECCC'], listings, 'ua')
    assert root.href == '/data/'
    assert [i.name for i in root.children] == ['NMME/', 'SubC/']
    nmme, subc = root.children
    assert nmme.children == []
    eccc, emc = subc.children
    assert emc.children == []
    assert [(i.href, i.current) for i in eccc.children] == [
        ('/data/SubC/ECCC/ua', True),
        ('/data/SubC/ECCC/tas', False),
    ]
    assert not eccc.current


def test_build_tree_marks_current_dataset():
    listings = [DisplayDataset(['SubC'], []), DisplayDataset(['ECCC'], [])]
    root = pages.build_tree('', ['SubC'], listings)
    assert root.href == '/'
    assert root.children[0].current
    assert not root.current


def test_build_tree_shows_hidden_dataset_on_path():
    # The root listing leaves out the hidden dataset S2S.
    listings = [DisplayDataset(['SubC'], []), DisplayDataset([], [])]
    root = pages.build_tree('/data', ['S2S'], listings)
    assert [i.name for i in root.children] == ['SubC/', 'S2S/']


def test_describe_values():
    assert pages.describe_values(np.array([850, 200, 100])) == '850, 200, 100'
    assert pages.describe_values(np.arange(-90.0, 91.0)) == '-90 to 90 by 1'
    assert pages.describe_values(np.array([0, 1, 2, 3, 4, 5, 10])) == '0 to 10'
    dates = np.array(['2024-06-13', '2024-06-17', '2024-06-20', '2024-06-24',
                      '2024-06-27', '2024-07-01', '2026-09-07'], dtype='datetime64[ns]')
    assert pages.describe_values(dates) == '2024-06-13 to 2026-09-07'
