import icechunk

import cataloging
import pydap_icechunk

icechunk.initialize_logs()
# TODO this is an important warning that needs to be addressed.
icechunk.set_logs_filter('[{message="The LocalFileSystem storage is not safe for concurrent commits}]=off')

dataset = cataloging.DatasetImpl('', None)
app = pydap_icechunk.Server(dataset)
