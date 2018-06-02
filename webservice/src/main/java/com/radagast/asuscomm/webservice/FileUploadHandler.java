package com.radagast.asuscomm.webservice;

import java.io.File;
import java.io.IOException;
import java.util.List;
import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import org.apache.commons.fileupload.FileItem;
import org.apache.commons.fileupload.disk.DiskFileItemFactory;
import org.apache.commons.fileupload.servlet.ServletFileUpload;

public class FileUploadHandler extends HttpServlet {

	private static final long serialVersionUID = 1L;
	private final String UPLOAD_DIRECTORY = MetaInfo.getPath("UPLOAD_DIRECTORY");
	private final String NON_IMAGE_PATH = MetaInfo.getPath("NON_IMAGE_PATH");
	private final String IMAGE_PATH = MetaInfo.getPath("IMAGE_PATH");
	
	@Override
	protected void doPost(HttpServletRequest request, HttpServletResponse response)
			throws ServletException, IOException {

		if (ServletFileUpload.isMultipartContent(request)) {
			try {
				List<FileItem> multiparts = new ServletFileUpload(new DiskFileItemFactory()).parseRequest(request);

				for (FileItem item : multiparts) {
					if (!item.isFormField()) {
						String fileName = item.getName();
						String mimeType = getServletContext().getMimeType(fileName);
						if (mimeType != null && mimeType.startsWith("image/")) {
							ImageHandler.handle(item, UPLOAD_DIRECTORY
									+ File.separator + IMAGE_PATH);
						} else {
							item.write(new File(UPLOAD_DIRECTORY + File.separator + 
									NON_IMAGE_PATH + File.separator + fileName));
						}
					}
				}

				request.setAttribute("message", "File Uploaded Successfully");
			} catch (Exception ex) {
				request.setAttribute("message", "File Upload Failed due to " + ex);
			}

		} else {
			request.setAttribute("message", "Sorry this Servlet only handles file upload request");
		}

		request.getRequestDispatcher("/UploadFile/result.jsp").forward(request, response);

	}

}