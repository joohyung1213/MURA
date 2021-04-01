from data import get_study_data, get_dataloaders


"""
데이터 경로 (-> 이미지)
CLASS
"""

if __name__ == '__main__':
    case_data = get_study_data('XR_ELBOW')
    phase_cat = ['train', 'valid']
    dataloaders = get_dataloaders(case_data, batch_size=1)

    print(case_data)
    """
    filename = 'MURA.csv'
    data.CuntomDataset(filename)
    """