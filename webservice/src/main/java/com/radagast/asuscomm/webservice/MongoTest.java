package com.radagast.asuscomm.webservice;

import java.io.File;
import java.io.IOException;
import java.util.concurrent.ThreadLocalRandom;

import org.apache.commons.io.FileUtils;
import org.bson.Document;
import org.bson.types.Binary;

import com.mongodb.MongoClient;
import com.mongodb.client.MongoCollection;
import com.mongodb.client.MongoDatabase;


public class MongoTest {
	public static void test1() {
		try (MongoClient mongoClient = new MongoClient( "localhost" , 27021 )) {
			MongoDatabase db = mongoClient.getDatabase("RasterData");
			
			MongoCollection<Document> collection = db.getCollection("RasterData");
			
			int randomNum = ThreadLocalRandom.current().nextInt(1, 100 + 1);
			
			for (int i = 0; i < 400000; ++i) {
				// Insert a binary data (byte array) into the database
				Document document = new Document("blob", "This is a byte array blob".getBytes());
				
				document.append("shard_key", randomNum);
				document.append("format", "txt");
				collection.insertOne(document);
			}
	
//			// Find and print the inserted byte array as String
//			for (Document doc : collection.find()) {
//			    Binary bin = doc.get("blob", org.bson.types.Binary.class);
//			    System.out.println(new String(bin.getData()));
//			}
		}
	}
	public static void main(String[] args) {
		test1();
//		try (MongoClient mongoClient = new MongoClient( "localhost" , 27021 )) {
//			MongoDatabase db = mongoClient.getDatabase("RasterData");
//			MongoCollection<Document> collection = db.getCollection("RasterData");
//
//			for (Document doc : collection.find()) {
//			    Binary bin = doc.get("blob", org.bson.types.Binary.class);
//			    String format = doc.getString("format");
//			    try {
//					FileUtils.writeByteArrayToFile(new File("D:\\Documents\\MongoTest\\image."+format), bin.getData());
//				} catch (IOException e) {
//					e.printStackTrace();
//				}
//			}
//		}
	}

}
