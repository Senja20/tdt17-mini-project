from modal import PoleDataset

if __name__ == '__main__':
    dataset = PoleDataset(
        '/cluster/projects/vc/data/ad/open/Poles/train/images',
        '/cluster/projects/vc/data/ad/open/Poles/train/labels',
    )
    print(len(dataset))
    sample = dataset[0]
    print(sample)
    print(sample.image)
    print(sample.annotations)
