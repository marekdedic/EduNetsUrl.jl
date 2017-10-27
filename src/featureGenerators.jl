export ngramFeatureGenerator, unigramFeatureGenerator, bigramFeatureGenerator, trigramFeatureGenerator;

"Generates an array of all the n-grams (substrings of length n) from a given string."
function ngrams(input::AbstractString, n::Int)::Vector{AbstractString}
	output = Vector{AbstractString}(max(length(input) - n + 1, 0));
	i = 1;
	j = 1;
	start = 1;
	stop = 1;
	while stop <= endof(input) && (j<n)
		stop = nextind(input,stop)
		j += 1;
	end

	while start <= endof(input)
		output[i] = input[start:min(stop,endof(input))];
		stop,start = nextind(input,stop),nextind(input,start)
i += 1
	end
	return output;
end

function ngramFeatureGenerator(input::AbstractString, modulo::Int, n::Int; T::DataType = Float32)::Vector{Float32}
	output = zeros(T, modulo);
	for i in ngrams(input, n)
		index = mod(hash(i), modulo);
		output[index + 1] += 1;
	end
	return output;
end

unigramFeatureGenerator(input::AbstractString, modulo::Int; T::DataType = Float32) = ngramFeatureGenerator(input, modulo, 1; T = T);
bigramFeatureGenerator(input::AbstractString, modulo::Int; T::DataType = Float32) = ngramFeatureGenerator(input, modulo, 2; T = T);
trigramFeatureGenerator(input::AbstractString, modulo::Int; T::DataType = Float32) = ngramFeatureGenerator(input, modulo, 3; T = T);
