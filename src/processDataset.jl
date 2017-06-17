"HEX decoder"
function decodeHEX(input::AbstractString)::AbstractString
	raw = input[5:end];
	splitted = split(raw, ":");
	raw = splitted[1];
	splitted = splitted[2:end];
	outputVec = Vector{Char}(cld(length(raw), 2));
	for i in 1:length(outputVec)
		outputVec[i] = Char(parse(Int, raw[(2i - 1):2i], 16));
	end
	output = AbstractString(outputVec);
	for i in splitted
		output *= i;
	end
	return output;
end

"Separates a given URL into 3 parts - domain, query, and path."
function separateUrl(url::AbstractString)::Tuple{Vector{AbstractString}, Vector{AbstractString}, Vector{AbstractString}}
	if contains(url, "://")
		url = split(url, "://")[2];
	end
	splitted = split(url, "/");
	rawDomain = splitted[1];
	if(startswith(rawDomain, "HEX"))
		domain = Vector{AbstractString}();
		push!(domain, decodeHEX(rawDomain));
	else
		domain = split(rawDomain, ".");
	end
	splitted = splitted[2:end];
	path = Vector{AbstractString}();
	query = Vector{AbstractString}();
	if length(splitted) != 0
		splitted2 = split(splitted[end], "?")
		splitted[end] = splitted2[1];
		if length(splitted2) > 1
			query = split(splitted2[2], "&");
		end
		path = splitted;
	end
	# Optional: add empty string when some part is empty array
	if(length(domain) == 0)
		push!(domain, "");
	end
	if(length(path) == 0)
		push!(path, "");
	end
	if(length(query) == 0)
		push!(query, "");
	end
	return (domain, path, query);
end

# Adds the vector v to a column of matrix a. If a is not of a sufficient size, it is extended
function addcolumn!(a::AbstractMatrix, v::AbstractVector, index::Int; step::Int = 1000)
	if size(a, 2) < index
		b = zeros(eltype(a), max(size(a, 1), length(v)), index + step);
		println("resizing $(size(a)) to $(size(b))");
		if !isempty(a)
			b[1:size(a, 1), 1:size(a, 2)] = a;
		end
		a = b;
	end
	a[:, index] = v;
	return a;
end

# Adds the vector v to a column of matrix a. If a is not of a sufficient size, it is extended
function additem!(a::AbstractVector, v, index::Int; step::Int = 1000)
	if length(a) < index
		b = zeros(eltype(a), index + step);
		if !isempty(a)
			b[1:length(a)] = a;
		end
		a = b;
	end
	a[index] = v;
	return a;
end

function processDataset(urls::Vector{AbstractString}, labels::Vector{Int}; featureCount::Int = 2053, featureGenerator::Function = trigramFeatureGenerator, T::DataType = Float32)::Dataset
	features = zeros(T,featureCount,8*length(urls))
	processedLabels = zeros(Int,8*length(urls));
	bags = zeros(Int,8*length(urls));
	urlParts = zeros(Int,8*length(urls));
	info = Vector{AbstractString}(0);
	freeidx=1;

	for j in 1:size(labels, 1)
		(domain, path, query) = separateUrl(urls[j]);
		for i in domain
			features = addcolumn!(features, featureGenerator(i, featureCount; T = T), freeidx, step = (div(freeidx, j) + 1) * (length(labels) - j + 1) + 1000);
			processedLabels = additem!(processedLabels, labels[j], freeidx, step = (div(freeidx, j) + 1) * (length(labels) - j + 1) + 1000);
			bags = additem!(bags, j, freeidx, step = (div(freeidx, j) + 1) * (length(labels) - j + 1) + 1000);
			urlParts = additem!(urlParts, 1, freeidx, step = (div(freeidx, j) + 1) * (length(labels) - j + 1) + 1000);
			freeidx += 1;
			push!(info, urls[j]);
		end
		for i in path
			features = addcolumn!(features, featureGenerator(i, featureCount; T = T), freeidx, step = (div(freeidx, j) + 1) * (length(labels) - j + 1) + 1000);
			processedLabels = additem!(processedLabels, labels[j], freeidx, step = (div(freeidx, j) + 1) * (length(labels) - j + 1) + 1000);
			bags = additem!(bags, j, freeidx, step = (div(freeidx, j) + 1) * (length(labels) - j + 1) + 1000);
			urlParts = additem!(urlParts, 2, freeidx, step = (div(freeidx, j) + 1) * (length(labels) - j + 1) + 1000);
			push!(info, urls[j]);
			freeidx += 1;
		end
		for i in query
			features = addcolumn!(features, featureGenerator(i, featureCount; T = T), freeidx, step = (div(freeidx, j) + 1) * (length(labels) - j + 1) + 1000);
			processedLabels = additem!(processedLabels, labels[j], freeidx, step = (div(freeidx, j) + 1) * (length(labels) - j + 1) + 1000);
			bags = additem!(bags, j, freeidx, step = (div(freeidx, j) + 1) * (length(labels) - j + 1) + 1000);
			urlParts = additem!(urlParts, 3, freeidx, step = (div(freeidx, j) + 1) * (length(labels)  - j + 1) + 1000);
			push!(info, urls[j]);
			freeidx += 1;
		end
	end
	return Dataset(features[:, 1:freeidx - 1], processedLabels[1:freeidx - 1], bags[1:freeidx - 1], urlParts[1:freeidx - 1]; info = info);
end
