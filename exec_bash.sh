docker exec -it `docker container ps|grep groundingdino|awk -F " " '{print $1}'` bash
