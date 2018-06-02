package com.radagast.asuscomm.webservice;

import java.awt.Graphics2D;
import java.awt.geom.AffineTransform;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

import javax.imageio.ImageIO;

import org.apache.commons.fileupload.FileItem;
import org.bson.Document;
import com.mongodb.MongoClient;
import com.mongodb.client.MongoCollection;
import com.mongodb.client.MongoDatabase;

public class ImageHandler {
	public static byte[] getBytes(BufferedImage bi, String format) {
		try(ByteArrayOutputStream baos = new ByteArrayOutputStream()) {
			ImageIO.write(bi, format, baos);
			baos.flush();
			return baos.toByteArray();
		} catch (IOException e) {
			System.out.println(e.getMessage());
			return null;
		}
	}
	
	public static void mongo(int shard_key, BufferedImage bi, String format) {
		byte[] blob = getBytes(bi, format);
		if (blob == null) return;
		try (MongoClient mongoClient = new MongoClient( "localhost" , 27021 )) {
			MongoDatabase db = mongoClient.getDatabase("RasterData");
			MongoCollection<Document> collection = db.getCollection("RasterData");

			Document document = new Document("blob", blob);
			document.append("shard_key", shard_key);
			document.append("format", format);
			
			collection.insertOne(document);
		}
	}
	
	public static BufferedImage scale(BufferedImage sbi) {
	    BufferedImage dbi = null;
	    if(sbi != null) {
	        dbi = new BufferedImage(sbi.getWidth() / 2, sbi.getHeight() / 2, sbi.getType());
	        Graphics2D g = dbi.createGraphics();
	        AffineTransform at = AffineTransform.getScaleInstance(0.5, 0.5);
	        g.drawRenderedImage(sbi, at);
	    }
	    return dbi;
	}
	
	public static List<IDIMG> make_mosaic(BufferedImage bi) {
		List<IDIMG> bis = new ArrayList<IDIMG>();
		int M = bi.getWidth() / MetaInfo.getMinWidth();
		int N = bi.getHeight() / MetaInfo.getMinHeight();
		int T = MetaInfo.getTileSize();
		int k = 0;
		for (int i = 0; i < M; ++i) {
			for (int j = 0; j < N; ++j) {
				BufferedImage sbi = bi.getSubimage(i * T, j * T, T, T);
				bis.add(new IDIMG(k++, sbi));
			}
			BufferedImage sbi = bi.getSubimage(i * T, N * T, T, bi.getHeight() - N * T);
			bis.add(new IDIMG(k++, sbi));
		}
		for (int j = 0; j < N; ++j) {
			BufferedImage sbi = bi.getSubimage(M * T, j * T, bi.getWidth() - M * T, T);
			bis.add(new IDIMG(k++, sbi));
		}
		BufferedImage sbi = bi.getSubimage(M * T, N * T, bi.getHeight() - N * T, bi.getHeight() - N * T);
		bis.add(new IDIMG(k++, sbi));
		return bis;
	}
	
	public static void handle(FileItem img, String upload_dir) throws IOException {
		InputStream in = new ByteArrayInputStream(img.get());
		BufferedImage bi = ImageIO.read(in);
		List<IDIMG> bis = make_mosaic(bi);
//		BufferedImage cropped_left = bi.getSubimage(0, 0, bi.getWidth()/2, bi.getHeight());
//		BufferedImage cropped_right = bi.getSubimage(bi.getWidth()/2, 0, bi.getWidth()/2, bi.getHeight());
		String format = MetaInfo.getPath("FORMAT");
		for (IDIMG p : bis) {
			mongo(p.getI(), p.getBi(), format);
		}
		while (bi.getWidth() > MetaInfo.getMinWidth() && bi.getHeight() > MetaInfo.getMinHeight()) {
			int M = (int) Math.round((bi.getWidth() / (float)MetaInfo.getMinWidth() + 0.5));
			int N = (int) Math.round((bi.getHeight() / (float)MetaInfo.getMinHeight() + 0.5));
			bi = scale(bi);
			bis = make_mosaic(bi);
			for (IDIMG p : bis) {
				mongo(M * N + p.getI(), p.getBi(), format);
			}
		}
//		mongo(1, cropped_left, format);
//		mongo(2, cropped_right, format);
//		ImageIO.write(bi,
//				format,
//				new File(upload_dir + File.separator +
//						MetaInfo.getPath("FULL_IMAGE_PATH") + File.separator +
//				img.getName().split("\\.")[0]	+ "." + format));
//		ImageIO.write(cropped_left,
//				format,
//				new File(upload_dir + File.separator +
//						MetaInfo.getPath("LEFT_PART_PATH") + File.separator +
//				img.getName().split("\\.")[0]	+ "." + format));
//		ImageIO.write(cropped_right,
//				format,
//				new File(upload_dir + File.separator +
//						MetaInfo.getPath("RIGHT_PART_PATH") + File.separator +
//				img.getName().split("\\.")[0]	+ "." + format));
	}
}

class IDIMG {
	public int getI() {
		return I;
	}
	public void setI(int i) {
		I = i;
	}
	public BufferedImage getBi() {
		return bi;
	}
	public void setBi(BufferedImage bi) {
		this.bi = bi;
	}
	public IDIMG(int i, BufferedImage bi) {
		super();
		I = i;
		this.bi = bi;
	}
	int I;
	BufferedImage bi;
}
