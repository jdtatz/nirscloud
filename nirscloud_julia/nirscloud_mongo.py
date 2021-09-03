import pymongo


def create_client(
    host='mongos.mongo.svc.cluster.local',
    port=27017,
    ssl=True,
    authSource="$external",
    authMechanism="MONGODB-X509",
    # Changed in version 3.12: ssl_certfile and ssl_keyfile were deprecated in favor of tlsCertificateKeyFile.
    ssl_certfile='/etc/mongo/jhub-keypem.pem',
    ssl_ca_certs='/etc/mongo/root-ca.pem',
    **extra_kwargs
) -> pymongo.MongoClient:
    return pymongo.MongoClient(
        host=host,
        port=port,
        ssl=ssl,
        authSource=authSource,
        authMechanism=authMechanism,
        ssl_certfile=ssl_certfile,
        ssl_ca_certs=ssl_ca_certs,
        **extra_kwargs
    )

core_schema_fields = "_id", "meta_id"
commmon_schema_fields = "group_id", "measurement_id", "note_id", "postfix_id", "session_id", "study_id", "subject_id", "the_date", "hdfs_path", "file_prefix"
fastrak_schema_fields = (*commmon_schema_fields,)
# *StartNanoTS and *EndNanoTS are not always there and aren't needed
nirs_schema_fields = (*commmon_schema_fields, "nirsDistances", "nirsWavelengths", "nirs_hz", "duration",) #"nirsStartNanoTS", "nirsEndNanoTS",)
dcs_schema_fields  = (*commmon_schema_fields, "dcsDistances", "dcsWavelength", "gains", "dcs_hz",) #"dcsStartNanoTS", "dcsEndNanoTS")


def query_meta(client: pymongo.MongoClient, query: dict, fields=None):
    db = client["meta"]
    col = db["meta3"]
    cursor = col.find(filter=query, projection=[f"{f}.val" for f in fields] if fields is not None else None)
    for doc in cursor:
        yield {k: (v["val"] if isinstance(v, dict) and "val" in v else v) for k, v in doc.items()}


def query_fastrak_meta(client: pymongo.MongoClient, query={}):
    return query_meta(client, {"n_fastrak_dedup.val": {"$exists": True}, **query}, fields=["meta_id", *fastrak_schema_fields])


def query_nirs_meta(client: pymongo.MongoClient, query={}):
    return query_meta(client, {"n_nirs_dedup.val": {"$exists": True}, **query}, fields=["meta_id", *nirs_schema_fields])
