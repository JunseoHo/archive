package archive.argument_resolver;

import org.springframework.core.MethodParameter;
import org.springframework.stereotype.Component;
import org.springframework.web.bind.support.WebDataBinderFactory;
import org.springframework.web.context.request.NativeWebRequest;
import org.springframework.web.method.support.HandlerMethodArgumentResolver;
import org.springframework.web.method.support.ModelAndViewContainer;

import archive.argument_resolver.argument.SampleAnnotation;
import archive.argument_resolver.argument.SampleArgument;

@Component
public class SampleArgumentResolver implements HandlerMethodArgumentResolver {

	/*
		MethodParameter 는 컨트롤러의 메서드가 가지는 매개변수에 하나에 대한 정보
	 */
	@Override
	public boolean supportsParameter(MethodParameter parameter) {
		return parameter.getParameterType().equals(SampleArgumentResolver.class)
			&& parameter.hasParameterAnnotation(SampleAnnotation.class);
	}

	@Override
	public Object resolveArgument(MethodParameter parameter, ModelAndViewContainer mavContainer,
		NativeWebRequest webRequest, WebDataBinderFactory binderFactory) throws Exception {
		String name = webRequest.getParameter("name");
		int age = Integer.parseInt(webRequest.getParameter("age"));
		return new SampleArgument(name, age);
	}
}
