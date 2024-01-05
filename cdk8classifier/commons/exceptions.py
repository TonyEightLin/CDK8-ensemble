class DataNotFoundError(Exception):
    def __init__(self, message):
        Exception.__init__(self, f'data not found: {message}')


class ProductionNotAppliedError(Exception):
    def __init__(self, message):
        Exception.__init__(self, f'this function does not apply for production data: {message}')


class RDKitSmilesParseError(Exception):
    def __init__(self, message):
        Exception.__init__(self, f'RDKit cannot parse this SMILES: {message}')
