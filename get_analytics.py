import os
import json
from urllib import parse
# Set environment variables
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'githubblog-412702-55c0c9eefcfd.json'

from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import DateRange
from google.analytics.data_v1beta.types import Dimension
from google.analytics.data_v1beta.types import Metric
from google.analytics.data_v1beta.types import RunReportRequest

property_id = 397192433

not_to_include = ['/','/about/','/posts/','/tags/']

def run_report(property_id):
    # Instantiates a client
    client = BetaAnalyticsDataClient()

    # Runs a request to get the number of users per page for the last 1 year

    request = RunReportRequest(
        property=f"properties/{property_id}",
        dimensions=[Dimension(name="pagePath")],
        metrics=[Metric(name="activeUsers")],
        date_ranges=[DateRange(start_date="2023-01-01", end_date="2023-12-31")]
    )

    response = client.run_report(request)
    
    # to json
    dict_response = {}

    for row in response.rows:
        link = row.dimension_values[0].value
        count = row.metric_values[0].value
        link_parsed = parse.unquote(link)
        
        if link_parsed not in not_to_include:
            dict_response[link_parsed] = count

    with open('analytics.json', 'w') as f:
        json.dump(dict_response, f)

    print("Finished")

# Run the report

run_report(property_id)
