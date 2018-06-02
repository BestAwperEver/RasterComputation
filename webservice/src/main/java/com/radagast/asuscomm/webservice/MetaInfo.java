package com.radagast.asuscomm.webservice;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import com.fasterxml.jackson.core.JsonParseException;
import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.ObjectMapper;

@SuppressWarnings("unchecked")
public final class MetaInfo {
	public static Map<String, Object> variables;
	
	static {
		try {
			variables = new ObjectMapper().readValue(
						new File("D:\\Documents\\eclipse-workspace\\webservice\\src\\main\\resources\\json\\config.json"),
						HashMap.class);
		} catch (JsonParseException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (JsonMappingException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public static String getPath(String to) {
		return variables.get(to).toString();
	}
	
	public static Map<String, Object> jsonToMap(JSONObject json) throws JSONException {
		Map<String, Object> retMap = new HashMap<String, Object>();
		if (json != JSONObject.NULL) {
			retMap = toMap(json);
		}
		return retMap;
	}
	private static Map<String, Object> toMap(JSONObject object) throws JSONException {
		Map<String, Object> map = new HashMap<String, Object>();

		Iterator<String> keysItr = object.keys();
		while (keysItr.hasNext()) {
			String key = keysItr.next();
			Object value = object.get(key);

			if (value instanceof JSONArray) {
				value = toList((JSONArray) value);
			}

			else if (value instanceof JSONObject) {
				value = toMap((JSONObject) value);
			}
			map.put(key, value);
		}
		return map;
	}
	private static List<Object> toList(JSONArray array) throws JSONException {
		List<Object> list = new ArrayList<Object>();
		for (int i = 0; i < array.length(); i++) {
			Object value = array.get(i);
			if (value instanceof JSONArray) {
				value = toList((JSONArray) value);
			}
			else if (value instanceof JSONObject) {
				value = toMap((JSONObject) value);
			} else if (value instanceof Integer) {
				value = (Integer) value;
			}
			list.add(value);
		}
		return list;
	}

	public static int getMinWidth() {
		return (int) (variables.get("MIN_W"));
	}

	public static int getMinHeight() {
		return (int) (variables.get("MIN_H"));
	}
	
	public static int getTileSize() {
		return (int) (variables.get("TILE_S"));
	}

}
