package archive.argument_resolver;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;

import archive.argument_resolver.argument.SampleAnnotation;
import archive.argument_resolver.argument.SampleArgument;

@Controller
public class MyController {

	@RequestMapping("/health_check")
	public void healthCheck(@SampleAnnotation SampleArgument sampleArgument) {
		System.out.println("healthCheck is called! -> " + sampleArgument);
	}

}
