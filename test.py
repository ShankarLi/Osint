# # import constants as ct
# from milvus_operations import MilvusOperations
# from pymilvus import connections, Collection, Partition, db


# milvus_ops = MilvusOperations(host=ct.HOST, port=ct.PORT, timeout=ct.TIMEOUT, db_name=ct.DATABASE)
# milvus_ops.connect_to_milvus()

# # results = milvus_ops.search_in_collection(
# #     collection_name="osint_test",
# #     query_embedding=ct.EMBEDDING_MODEL.encode("China Electronics Technology Group Corporation"),
# #     top_k=100
# # )
# # print(results)

# # db.using_database("rfa_cbt_vdb")

# collection = Collection("osint_test")

# # iterator = collection.query_iterator(
# #     batch_size=10,
# #     output_fields=["id","text"]  # Specify the fields you want to see
# # )

# # results = []

# # while True:
# #     result = iterator.next()
# #     if not result:
# #         iterator.close()
# #         break
# #     print(result)
# #     results += result
# # print(results)

# # partition = Partition(collection, name="test_collection")

# res = collection.query(
#     expr="id in [100]",
#     output_fields=["id", "text"],
# )
# print(res)

import os 

print(os.cpu_count())