using EduNetsUrl;

function testGenerator(generator::Function)::Bool
	size = rand(1:1000);
	len = rand(1:10000);
	str = randstring(len);
	hash = generator(str, size);
	for i in 1:100
		if generator(str, size) != hash
			return false;
		end
	end
	return true;
end
