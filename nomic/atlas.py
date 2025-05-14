# Ajustes realizados:
# - Confirmación de que no haya restricciones en la edición de proyectos.

class Project:
    def __init__(self, name, description, colorable_fields,
                 reset_project_if_exists, add_datums_if_exists,
                 projection_n_neighbors, projection_epochs):
        self.name = name
        self.description = description
        self.colorable_fields = colorable_fields
        self.reset_project_if_exists = reset_project_if_exists
        self.add_datums_if_exists = add_datums_if_exists
        self.projection_n_neighbors = projection_n_neighbors
        self.projection_epochs = projection_epochs

    def create_project(self):
        # Logic to create a project
        pass

    def reset_project(self):
        if self.reset_project_if_exists:
            # Logic to reset the project
            pass

    def add_data_to_project(self):
        if self.add_datums_if_exists:
            # Logic to add data to the project
            pass

    def build_projection(self):
        # Logic to build projection using n_neighbors and epochs
        pass
