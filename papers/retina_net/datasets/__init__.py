from torch.utils.data import Dataset


class ObjectDetectionDataset(Dataset):

    def __init__(self, data_path, schema_path):
        self.data_path = data_path
        self.schema_path = schema_path
        self.image_to_annotations = {}
        self.class_to_id = {}
        self.image_to_anchors = {}

        # parse CSV data and create appropriate mappings.
        self.read_annotations()
        self.read_schema()

        self.image_paths = list(self.image_to_annotations.keys())
        self.num_images = len(self.image_to_annotations)
        self.num_classes = len(self.class_to_id)

    def read_annotations(self):
        """Function to parse data csv and create image to annotations mapping"""

        try:
            with open(self.data_path) as f:
                annotations = f.readlines()
        except ValueError as e:
            raise ValueError('invalid CSV data file: {}, {}'.format(self.data_path, e))


        for annotation_line in annotations:
            try:
                image_path, x1, y1, x2, y2, class_label = annotation_line.split(',')
                class_label = class_label.strip('\n')
            except ValueError as e:
                raise ValueError('invalid annotation: {}, {}'.format(annotation_line, e))

            if image_path not in self.image_to_annotations:
                self.image_to_annotations[image_path] = []

            try:
                annotation = {}
                annotation['box']['x1'] = int(x1)
                annotation['box']['y1'] = int(y1)
                annotation['box']['x2'] = int(x2)
                annotation['box']['y2'] = int(y2)
                annotation['class'] = class_label
                self.image_to_annotations.append(annotation)

            except ValueError as e:
                raise ValueError('invalid annotation: {},{},{},{},{}'.format(x1, y1, x2, y2, e))

    def read_schema(self):
        """Function to parse schema csv and create class to id mapping"""

        try:
            with open(self.schema_path) as f:
                schema = f.readlines()
        except ValueError as e:
            raise ValueError('invalid schema file: {}, {}'.format(self.schema_path), e)


        for schema_line in schema:
            try:
                class_label, _id = schema_line.split(',')
                _id = int(_id.strip('\n'))
            except ValueError as e:
                raise ValueError('invalid class label or id: {}, {}'.format(schema_line, e))

            if class_label not in self.class_to_id:
                self.class_to_id[class_label] = _id


    def __len__(self):
        return len(self.image_to_annotations)


    def __getitem__(self, index):

        image_path = self.image_paths[index]



        return