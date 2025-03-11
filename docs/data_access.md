# Data Access

Here is the data source that you can connect with your SQL queries:

- Ms SQL Server
- PostgreSQL
- AWS RedShift
- Google BigQuery
- .csv
- .json
- pickle
- parquet


## Connection PostgreSQL - MS SQL - AWS RedShift

        data_source = "postgresql"
        connector = {"user": ***, "password": ***, "server": "127.0.0.1",
                     "port": ****, "db": ***}
        data_main_path ="""
                           SELECT
                            customer_indicator,
                            amount_indicator,
                            time_indicator,
                           FROM table
                       """
## Connection Google BigQuery

        data_source = "googlebigquery"
        connector = {"data_main_path": "./json_file_where_you_stored",
                     "db": "flash-clover-*********.json"}
        data_main_path ="""
                   SELECT
                        customer_indicator,
                        amount_indicator,
                        time_indicator,
                    FROM tablee
               """

## Connection csv - .json - .pickle - .parquet

        data_source = "csv"
        data_main_path = "./data_where_you_store/***.csv"

